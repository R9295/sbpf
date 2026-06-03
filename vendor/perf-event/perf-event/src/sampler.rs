use std::convert::TryInto;
use std::fs::File;
use std::io;
use std::os::fd::{AsRawFd, RawFd};
use std::ptr::{copy_nonoverlapping, null_mut, read_volatile, write_volatile};
use std::sync::atomic::{fence, Ordering};

use crate::check_errno_syscall;
use crate::flags::SampleFlag;
use crate::sys;

/// One parsed `PERF_RECORD_SAMPLE` record.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SampleRecord {
    /// Sampled instruction pointer.
    pub ip: Option<u64>,
    /// Sampled process id.
    pub pid: Option<u32>,
    /// Sampled thread id.
    pub tid: Option<u32>,
    /// Sample timestamp, in the clock selected for the event.
    pub time: Option<u64>,
    /// Sampled address, for events that provide one.
    pub addr: Option<u64>,
    /// Event id.
    pub id: Option<u64>,
    /// Event stream id.
    pub stream_id: Option<u64>,
    /// Sampled CPU.
    pub cpu: Option<u32>,
    /// Sampling period.
    pub period: Option<u64>,
}

/// A perf event backed by an mmap sample ring.
///
/// Build one with [`Builder::build_sampler`](crate::Builder::build_sampler).
pub struct Sampler {
    file: File,
    id: u64,
    sample_type: SampleFlag,
    mapping: *mut libc::c_void,
    mapping_len: usize,
    data: *mut u8,
    data_size: usize,
    tail: u64,
}

struct RawRecord {
    record_type: u32,
    data: Vec<u8>,
}

impl Sampler {
    pub(crate) fn new(
        file: File,
        id: u64,
        sample_type: u64,
        data_pages: usize,
    ) -> io::Result<Self> {
        let Some(sample_type) = SampleFlag::from_bits(sample_type) else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "unsupported sample fields requested",
            ));
        };
        if data_pages == 0 || !data_pages.is_power_of_two() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "sample data pages must be a non-zero power of two",
            ));
        }

        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page_size <= 0 {
            return Err(io::Error::last_os_error());
        }
        let page_size = page_size as usize;
        let mapping_len = page_size
            .checked_mul(data_pages.saturating_add(1))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "mmap size overflow"))?;
        let mapping = unsafe {
            libc::mmap(
                null_mut(),
                mapping_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if mapping == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        let metadata = mapping.cast::<sys::bindings::perf_event_mmap_page>();
        let data_offset = unsafe { (*metadata).data_offset as usize };
        let data_size = unsafe { (*metadata).data_size as usize };
        let tail = unsafe { read_volatile(&(*metadata).data_tail) };
        if data_offset == 0
            || data_size == 0
            || !data_size.is_power_of_two()
            || data_offset.saturating_add(data_size) > mapping_len
        {
            unsafe {
                libc::munmap(mapping, mapping_len);
            }
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid perf sample mmap metadata",
            ));
        }

        Ok(Self {
            file,
            id,
            sample_type,
            mapping,
            mapping_len,
            data: unsafe { mapping.cast::<u8>().add(data_offset) },
            data_size,
            tail,
        })
    }

    /// Return this sampler's kernel-assigned unique id.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Allow this sampler to begin recording events.
    pub fn enable(&mut self) -> io::Result<()> {
        check_errno_syscall(|| unsafe { sys::ioctls::ENABLE(self.file.as_raw_fd(), 0) }).map(|_| ())
    }

    /// Make this sampler stop recording events.
    pub fn disable(&mut self) -> io::Result<()> {
        check_errno_syscall(|| unsafe { sys::ioctls::DISABLE(self.file.as_raw_fd(), 0) })
            .map(|_| ())
    }

    /// Reset this sampler's event count to zero.
    pub fn reset(&mut self) -> io::Result<()> {
        check_errno_syscall(|| unsafe { sys::ioctls::RESET(self.file.as_raw_fd(), 0) }).map(|_| ())
    }

    /// Return the next available sample record, if any.
    ///
    /// Non-sample records are skipped.
    pub fn next_sample(&mut self) -> io::Result<Option<SampleRecord>> {
        while let Some(record) = self.next_record()? {
            if record.record_type == sys::bindings::PERF_RECORD_SAMPLE {
                return self.parse_sample(&record.data).map(Some);
            }
        }
        Ok(None)
    }

    fn next_record(&mut self) -> io::Result<Option<RawRecord>> {
        let metadata = self.mapping.cast::<sys::bindings::perf_event_mmap_page>();
        let head = unsafe { read_volatile(&(*metadata).data_head) };
        fence(Ordering::Acquire);
        if self.tail == head {
            return Ok(None);
        }
        if head < self.tail {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "perf sample ring head moved backwards",
            ));
        }
        let available = head - self.tail;
        if available < std::mem::size_of::<sys::bindings::perf_event_header>() as u64 {
            return Ok(None);
        }

        let mut header = [0u8; std::mem::size_of::<sys::bindings::perf_event_header>()];
        self.copy_from_ring(self.tail, &mut header);
        let record_type = u32::from_ne_bytes(header[0..4].try_into().unwrap());
        let size = u16::from_ne_bytes(header[6..8].try_into().unwrap()) as usize;
        if size < header.len() || size as u64 > available {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid perf sample record size",
            ));
        }

        let mut data = vec![0; size];
        self.copy_from_ring(self.tail, &mut data);
        self.tail = self.tail.saturating_add(size as u64);
        unsafe { write_volatile(&mut (*metadata).data_tail, self.tail) };
        fence(Ordering::Release);

        Ok(Some(RawRecord { record_type, data }))
    }

    fn copy_from_ring(&self, start: u64, out: &mut [u8]) {
        let offset = start as usize & (self.data_size - 1);
        let first_len = out.len().min(self.data_size - offset);
        unsafe {
            copy_nonoverlapping(self.data.add(offset), out.as_mut_ptr(), first_len);
            if first_len < out.len() {
                copy_nonoverlapping(
                    self.data,
                    out.as_mut_ptr().add(first_len),
                    out.len() - first_len,
                );
            }
        }
    }

    fn parse_sample(&self, data: &[u8]) -> io::Result<SampleRecord> {
        let mut cursor = std::mem::size_of::<sys::bindings::perf_event_header>();
        let mut sample = SampleRecord::default();

        if self.sample_type.contains(SampleFlag::IP) {
            sample.ip = Some(read_u64(data, &mut cursor)?);
        }
        if self.sample_type.contains(SampleFlag::TID) {
            sample.pid = Some(read_u32(data, &mut cursor)?);
            sample.tid = Some(read_u32(data, &mut cursor)?);
        }
        if self.sample_type.contains(SampleFlag::TIME) {
            sample.time = Some(read_u64(data, &mut cursor)?);
        }
        if self.sample_type.contains(SampleFlag::ADDR) {
            sample.addr = Some(read_u64(data, &mut cursor)?);
        }
        if self.sample_type.contains(SampleFlag::ID) {
            sample.id = Some(read_u64(data, &mut cursor)?);
        }
        if self.sample_type.contains(SampleFlag::STREAM_ID) {
            sample.stream_id = Some(read_u64(data, &mut cursor)?);
        }
        if self.sample_type.contains(SampleFlag::CPU) {
            sample.cpu = Some(read_u32(data, &mut cursor)?);
            let _reserved = read_u32(data, &mut cursor)?;
        }
        if self.sample_type.contains(SampleFlag::PERIOD) {
            sample.period = Some(read_u64(data, &mut cursor)?);
        }

        Ok(sample)
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.mapping, self.mapping_len);
        }
    }
}

impl std::fmt::Debug for Sampler {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            fmt,
            "Sampler {{ fd: {}, id: {}, data_size: {} }}",
            self.file.as_raw_fd(),
            self.id,
            self.data_size
        )
    }
}

impl AsRawFd for Sampler {
    fn as_raw_fd(&self) -> RawFd {
        self.file.as_raw_fd()
    }
}

fn read_u64(data: &[u8], cursor: &mut usize) -> io::Result<u64> {
    let end = cursor.saturating_add(8);
    let bytes = data.get(*cursor..end).ok_or_else(|| {
        io::Error::new(io::ErrorKind::UnexpectedEof, "truncated perf sample record")
    })?;
    *cursor = end;
    Ok(u64::from_ne_bytes(bytes.try_into().unwrap()))
}

fn read_u32(data: &[u8], cursor: &mut usize) -> io::Result<u32> {
    let end = cursor.saturating_add(4);
    let bytes = data.get(*cursor..end).ok_or_else(|| {
        io::Error::new(io::ErrorKind::UnexpectedEof, "truncated perf sample record")
    })?;
    *cursor = end;
    Ok(u32::from_ne_bytes(bytes.try_into().unwrap()))
}
