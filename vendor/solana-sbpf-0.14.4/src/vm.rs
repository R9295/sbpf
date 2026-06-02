#![allow(clippy::arithmetic_side_effects)]
// Derived from uBPF <https://github.com/iovisor/ubpf>
// Copyright 2015 Big Switch Networks, Inc
//      (uBPF: VM architecture, parts of the interpreter, originally in C)
// Copyright 2016 6WIND S.A. <quentin.monnet@6wind.com>
//      (Translation to Rust, MetaBuff/multiple classes addition, hashmaps for syscalls)
// Copyright 2020 Solana Maintainers <maintainers@solana.com>
//
// Licensed under the Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> or
// the MIT license <http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Virtual machine for eBPF programs.

use crate::{
    ebpf,
    elf::Executable,
    error::{EbpfError, ProgramResult},
    interpreter::Interpreter,
    memory_region::MemoryMapping,
    program::{BuiltinFunction, BuiltinProgram, FunctionRegistry, SBPFVersion},
    static_analysis::{
        Analysis, DummyContextObject, RegisterTraceEntry, SyscallAllocationSnapshot,
        SyscallTraceEntry,
    },
};
use std::{collections::BTreeMap, fmt::Debug, mem::offset_of, ptr};

#[cfg(feature = "shuttle-test")]
use shuttle::sync::Arc;
#[cfg(not(feature = "shuttle-test"))]
use std::sync::Arc;

#[cfg(all(feature = "jit", not(feature = "shuttle-test")))]
use rand::{thread_rng, Rng};
#[cfg(all(feature = "jit", feature = "shuttle-test"))]
use shuttle::rand::{thread_rng, Rng};

/// Function used to sample host allocation counters for syscall tracing.
pub type SyscallAllocationSampler = fn() -> SyscallAllocationSnapshot;

static SYSCALL_ALLOCATION_SAMPLER: std::sync::RwLock<Option<SyscallAllocationSampler>> =
    std::sync::RwLock::new(None);

/// Installs the host allocation sampler used by syscall tracing.
pub fn set_syscall_allocation_sampler(sampler: SyscallAllocationSampler) {
    *SYSCALL_ALLOCATION_SAMPLER.write().unwrap() = Some(sampler);
}

fn read_syscall_allocation_snapshot() -> Option<SyscallAllocationSnapshot> {
    let sampler = *SYSCALL_ALLOCATION_SAMPLER.read().unwrap();
    sampler.map(|sampler| sampler())
}

/// Shift the RUNTIME_ENVIRONMENT_KEY by this many bits to the LSB
///
/// 3 bits for 8 Byte alignment, and 1 bit to have encoding space for the RuntimeEnvironment.
#[cfg(feature = "jit")]
const PROGRAM_ENVIRONMENT_KEY_SHIFT: u32 = 4;
#[cfg(feature = "jit")]
static RUNTIME_ENVIRONMENT_KEY: std::sync::OnceLock<i32> = std::sync::OnceLock::<i32>::new();

/// Returns (and if not done before generates) the encryption key for the VM pointer
pub fn get_runtime_environment_key() -> i32 {
    #[cfg(feature = "jit")]
    {
        *RUNTIME_ENVIRONMENT_KEY
            .get_or_init(|| thread_rng().gen::<i32>() >> PROGRAM_ENVIRONMENT_KEY_SHIFT)
    }
    #[cfg(not(feature = "jit"))]
    0
}

/// VM configuration settings
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// Maximum call depth
    pub max_call_depth: usize,
    /// Size of a stack frame in bytes, must match the size specified in the LLVM BPF backend
    pub stack_frame_size: usize,
    /// Enables the use of MemoryMapping and MemoryRegion for address translation
    pub enable_address_translation: bool,
    /// Enables gaps in VM address space between the stack frames
    pub enable_stack_frame_gaps: bool,
    /// Maximal pc distance after which a new instruction meter validation is emitted by the JIT
    pub instruction_meter_checkpoint_distance: usize,
    /// Enable instruction meter and limiting
    pub enable_instruction_meter: bool,
    /// Enable instruction tracing
    pub enable_register_tracing: bool,
    /// Enable syscall-level tracing
    pub enable_syscall_tracing: bool,
    /// Enable dynamic string allocation for labels
    pub enable_symbol_and_section_labels: bool,
    /// Reject ELF files containing issues that the verifier did not catch before (up to v0.2.21)
    pub reject_broken_elfs: bool,
    #[cfg(feature = "jit")]
    /// Ratio of native host instructions per random no-op in JIT (0 = OFF)
    pub noop_instruction_rate: u32,
    #[cfg(feature = "jit")]
    /// Enable disinfection of immediate values and offsets provided by the user in JIT
    pub sanitize_user_provided_values: bool,
    /// Avoid copying read only sections when possible
    pub optimize_rodata: bool,
    /// Allow a memory region at age zero in the aligned memory mapping
    pub allow_memory_region_zero: bool,
    /// Use aligned memory mapping
    pub aligned_memory_mapping: bool,
    /// Allowed [SBPFVersion]s
    pub enabled_sbpf_versions: std::ops::RangeInclusive<SBPFVersion>,
}

impl Config {
    /// Returns the size of the stack memory region
    pub fn stack_size(&self) -> usize {
        self.stack_frame_size * self.max_call_depth
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_call_depth: 64,
            stack_frame_size: 4_096,
            enable_address_translation: true,
            enable_stack_frame_gaps: true,
            instruction_meter_checkpoint_distance: 10000,
            enable_instruction_meter: true,
            enable_register_tracing: false,
            enable_syscall_tracing: false,
            enable_symbol_and_section_labels: false,
            reject_broken_elfs: false,
            #[cfg(feature = "jit")]
            noop_instruction_rate: 256,
            #[cfg(feature = "jit")]
            sanitize_user_provided_values: true,
            optimize_rodata: true,
            allow_memory_region_zero: true,
            aligned_memory_mapping: false,
            enabled_sbpf_versions: SBPFVersion::V0..=SBPFVersion::V4,
        }
    }
}

/// Static constructors for Executable
impl<C: ContextObject> Executable<C> {
    /// Creates an executable from an ELF file
    pub fn from_elf(elf_bytes: &[u8], loader: Arc<BuiltinProgram<C>>) -> Result<Self, EbpfError> {
        let executable = Executable::load(elf_bytes, loader)?;
        Ok(executable)
    }
    /// Creates an executable from machine code
    pub fn from_text_bytes(
        text_bytes: &[u8],
        loader: Arc<BuiltinProgram<C>>,
        sbpf_version: SBPFVersion,
        function_registry: FunctionRegistry<usize>,
    ) -> Result<Self, EbpfError> {
        Executable::new_from_text_bytes(text_bytes, loader, sbpf_version, function_registry)
            .map_err(EbpfError::ElfError)
    }
}

/// Runtime context
pub trait ContextObject {
    /// Consume instructions from meter
    fn consume(&mut self, amount: u64);
    /// Get the number of remaining instructions allowed
    fn get_remaining(&self) -> u64;
}

/// Statistic of taken branches (from a recorded trace)
pub struct DynamicAnalysis {
    /// Maximal edge counter value
    pub edge_counter_max: usize,
    /// src_node, dst_node, edge_counter
    pub edges: BTreeMap<usize, BTreeMap<usize, usize>>,
}

impl DynamicAnalysis {
    /// Accumulates a trace
    pub fn new(register_trace: &[[u64; 12]], analysis: &Analysis) -> Self {
        let mut result = Self {
            edge_counter_max: 0,
            edges: BTreeMap::new(),
        };
        let mut last_basic_block = usize::MAX;
        for traced_instruction in register_trace.iter() {
            let pc = traced_instruction[11] as usize;
            if analysis.cfg_nodes.contains_key(&pc) {
                let counter = result
                    .edges
                    .entry(last_basic_block)
                    .or_default()
                    .entry(pc)
                    .or_insert(0);
                *counter += 1;
                result.edge_counter_max = result.edge_counter_max.max(*counter);
                last_basic_block = pc;
            }
        }
        result
    }
}

/// A call frame used for function calls inside the Interpreter
#[derive(Clone, Default)]
pub struct CallFrame {
    /// The caller saved registers
    pub caller_saved_registers: [u64; ebpf::SCRATCH_REGS],
    /// The callers frame pointer
    pub frame_pointer: u64,
    /// The target_pc of the exit instruction which returns back to the caller
    pub target_pc: u64,
}

/// Indices of slots inside [EbpfVm]
pub enum RuntimeEnvironmentSlot {
    /// [EbpfVm::host_stack_pointer]
    HostStackPointer = offset_of!(EbpfVm<DummyContextObject>, host_stack_pointer) as isize,
    /// [EbpfVm::call_depth]
    CallDepth = offset_of!(EbpfVm<DummyContextObject>, call_depth) as isize,
    /// [EbpfVm::context_object_pointer]
    ContextObjectPointer = offset_of!(EbpfVm<DummyContextObject>, context_object_pointer) as isize,
    /// [EbpfVm::previous_instruction_meter]
    PreviousInstructionMeter =
        offset_of!(EbpfVm<DummyContextObject>, previous_instruction_meter) as isize,
    /// [EbpfVm::due_insn_count]
    DueInsnCount = offset_of!(EbpfVm<DummyContextObject>, due_insn_count) as isize,
    /// [EbpfVm::stopwatch_numerator]
    StopwatchNumerator = offset_of!(EbpfVm<DummyContextObject>, stopwatch_numerator) as isize,
    /// [EbpfVm::stopwatch_denominator]
    StopwatchDenominator = offset_of!(EbpfVm<DummyContextObject>, stopwatch_denominator) as isize,
    /// [EbpfVm::registers]
    Registers = offset_of!(EbpfVm<DummyContextObject>, registers) as isize,
    /// [EbpfVm::program_result]
    ProgramResult = offset_of!(EbpfVm<DummyContextObject>, program_result) as isize,
    /// [EbpfVm::memory_mapping]
    MemoryMapping = offset_of!(EbpfVm<DummyContextObject>, memory_mapping) as isize,
    /// [EbpfVm::register_trace]
    RegisterTrace = offset_of!(EbpfVm<DummyContextObject>, register_trace) as isize,
    /// [EbpfVm::syscall_trace_pc]
    SyscallTracePc = offset_of!(EbpfVm<DummyContextObject>, syscall_trace_pc) as isize,
    /// [EbpfVm::syscall_trace_hash]
    SyscallTraceHash = offset_of!(EbpfVm<DummyContextObject>, syscall_trace_hash) as isize,
}

/// A virtual machine to run eBPF programs.
///
/// # Examples
///
/// ```
/// use solana_sbpf::{
///     aligned_memory::AlignedMemory,
///     ebpf,
///     elf::Executable,
///     memory_region::{MemoryMapping, MemoryRegion},
///     program::{BuiltinProgram, FunctionRegistry, SBPFVersion},
///     verifier::RequisiteVerifier,
///     vm::{Config, EbpfVm},
/// };
/// use test_utils::TestContextObject;
///
/// let prog = &[
///     0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // add64 r0, 0
///     0x95, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00  // exit
/// ];
/// let mem = &mut [
///     0xaa, 0xbb, 0x11, 0x22, 0xcc, 0xdd
/// ];
///
/// let loader = std::sync::Arc::new(BuiltinProgram::new_mock());
/// let function_registry = FunctionRegistry::default();
/// let mut executable = Executable::<TestContextObject>::from_text_bytes(prog, loader.clone(), SBPFVersion::V4, function_registry).unwrap();
/// executable.verify::<RequisiteVerifier>().unwrap();
/// let mut context_object = TestContextObject::new(2);
/// let sbpf_version = executable.get_sbpf_version();
///
/// let mut stack = AlignedMemory::<{ebpf::HOST_ALIGN}>::zero_filled(executable.get_config().stack_size());
/// let stack_len = stack.len();
/// let mut heap = AlignedMemory::<{ebpf::HOST_ALIGN}>::with_capacity(0);
///
/// let regions: Vec<MemoryRegion> = vec![
///     executable.get_ro_region(),
///     MemoryRegion::new_writable(
///         stack.as_slice_mut(),
///         ebpf::MM_STACK_START,
///     ),
///     MemoryRegion::new_writable(heap.as_slice_mut(), ebpf::MM_HEAP_START),
///     MemoryRegion::new_writable(mem, ebpf::MM_INPUT_START),
/// ];
///
/// let memory_mapping = MemoryMapping::new(regions, executable.get_config(), sbpf_version).unwrap();
///
/// let mut vm = EbpfVm::new(loader, sbpf_version, &mut context_object, memory_mapping, stack_len);
///
/// let (instruction_count, result) = vm.execute_program(&executable, true);
/// assert_eq!(instruction_count, 2);
/// assert_eq!(result.unwrap(), 0);
/// ```
#[repr(C)]
pub struct EbpfVm<'a, C: ContextObject> {
    /// Needed to exit from the guest back into the host
    pub host_stack_pointer: *mut u64,
    /// The current call depth.
    ///
    /// Incremented on calls and decremented on exits. It's used to enforce
    /// config.max_call_depth and to know when to terminate execution.
    pub call_depth: u64,
    /// Pointer to ContextObject
    pub context_object_pointer: &'a mut C,
    /// Last return value of instruction_meter.get_remaining()
    pub previous_instruction_meter: u64,
    /// Outstanding value to instruction_meter.consume()
    pub due_insn_count: u64,
    /// CPU cycles accumulated by the stop watch
    pub stopwatch_numerator: u64,
    /// Number of times the stop watch was used
    pub stopwatch_denominator: u64,
    /// Registers inlined
    pub registers: [u64; 12],
    /// ProgramResult inlined
    pub program_result: ProgramResult,
    /// MemoryMapping inlined
    pub memory_mapping: MemoryMapping,
    /// Stack of CallFrames used by the Interpreter
    pub call_frames: Vec<CallFrame>,
    /// Loader built-in program
    pub loader: Arc<BuiltinProgram<C>>,
    /// Collector for the instruction trace
    pub register_trace: Vec<RegisterTraceEntry>,
    /// Collector for syscall-level native instruction traces
    pub syscall_trace: Vec<SyscallTraceEntry>,
    /// Program counter of the syscall currently dispatched by the JIT
    pub(crate) syscall_trace_pc: u64,
    /// Hash of the syscall currently dispatched by the JIT
    pub(crate) syscall_trace_hash: u64,
    syscall_instruction_counter: SyscallInstructionCounter,
    /// TCP port for the debugger interface
    #[cfg(feature = "debugger")]
    pub debug_port: Option<u16>,
    /// Debug metadata passed
    #[cfg(feature = "debugger")]
    pub debug_metadata: Option<String>,
}

impl<'a, C: ContextObject> EbpfVm<'a, C> {
    /// Creates a new virtual machine instance.
    pub fn new(
        loader: Arc<BuiltinProgram<C>>,
        sbpf_version: SBPFVersion,
        context_object: &'a mut C,
        mut memory_mapping: MemoryMapping,
        stack_len: usize,
    ) -> Self {
        let config = loader.get_config();
        let mut registers = [0u64; 12];
        registers[ebpf::FRAME_PTR_REG] =
            ebpf::MM_STACK_START.saturating_add(if !sbpf_version.manual_stack_frame_bump() {
                config.stack_frame_size
            } else {
                stack_len
            } as u64);
        if !config.enable_address_translation {
            memory_mapping = MemoryMapping::new_identity();
        }
        EbpfVm {
            host_stack_pointer: std::ptr::null_mut(),
            call_depth: 0,
            context_object_pointer: context_object,
            previous_instruction_meter: 0,
            due_insn_count: 0,
            stopwatch_numerator: 0,
            stopwatch_denominator: 0,
            registers,
            program_result: ProgramResult::Ok(0),
            memory_mapping,
            call_frames: vec![CallFrame::default(); config.max_call_depth],
            loader,
            #[cfg(feature = "debugger")]
            debug_port: std::env::var("VM_DEBUG_PORT")
                .ok()
                .and_then(|v| v.parse::<u16>().ok()),
            #[cfg(feature = "debugger")]
            debug_metadata: None,
            register_trace: Vec::default(),
            syscall_trace: Vec::default(),
            syscall_trace_pc: 0,
            syscall_trace_hash: 0,
            syscall_instruction_counter: SyscallInstructionCounter::default(),
        }
    }

    /// Execute the program
    ///
    /// If interpreted = `false` then the JIT compiled executable is used.
    pub fn execute_program(
        &mut self,
        executable: &Executable<C>,
        interpreted: bool,
    ) -> (u64, ProgramResult) {
        debug_assert!(Arc::ptr_eq(&self.loader, executable.get_loader()));
        self.registers[11] = executable.get_entrypoint_instruction_offset() as u64;
        let config = executable.get_config();
        let initial_insn_count = self.context_object_pointer.get_remaining();
        self.previous_instruction_meter = initial_insn_count;
        self.due_insn_count = 0;
        self.program_result = ProgramResult::Ok(0);
        if interpreted {
            #[cold]
            #[inline(never)]
            #[cfg(feature = "debugger")]
            fn run_interpreter<C: ContextObject>(mut interpreter: Interpreter<C>) {
                let debug_port = interpreter.vm.debug_port.clone();
                if let Some(debug_port) = debug_port {
                    crate::debugger::execute(&mut interpreter, debug_port);
                } else {
                    while interpreter.step() {}
                }
            }
            #[cold]
            #[inline(never)]
            #[cfg(not(feature = "debugger"))]
            fn run_interpreter<C: ContextObject>(mut interpreter: Interpreter<C>) {
                while interpreter.step() {}
            }
            let interpreter = Interpreter::new(self, executable, self.registers);
            run_interpreter(interpreter);
        } else {
            #[cfg(all(feature = "jit", not(target_os = "windows"), target_arch = "x86_64"))]
            {
                let compiled_program = match executable
                    .get_compiled_program()
                    .ok_or_else(|| EbpfError::JitNotCompiled)
                {
                    Ok(compiled_program) => compiled_program,
                    Err(error) => return (0, ProgramResult::Err(error)),
                };
                compiled_program.invoke(config, self, self.registers);
            }
            #[cfg(not(all(feature = "jit", not(target_os = "windows"), target_arch = "x86_64")))]
            {
                return (0, ProgramResult::Err(EbpfError::JitNotCompiled));
            }
        };
        let instruction_count = if config.enable_instruction_meter {
            self.context_object_pointer.consume(self.due_insn_count);
            initial_insn_count.saturating_sub(self.context_object_pointer.get_remaining())
        } else {
            0
        };
        let mut result = ProgramResult::Ok(0);
        std::mem::swap(&mut result, &mut self.program_result);
        (instruction_count, result)
    }

    /// Invokes a built-in function
    pub fn invoke_function(&mut self, function: BuiltinFunction<C>) {
        function(
            unsafe {
                std::ptr::addr_of_mut!(*self)
                    .cast::<u64>()
                    .offset(get_runtime_environment_key() as isize)
                    .cast::<Self>()
            },
            self.registers[1],
            self.registers[2],
            self.registers[3],
            self.registers[4],
            self.registers[5],
        );
    }

    /// Invokes a built-in function and records syscall-level native instruction count.
    pub(crate) fn invoke_function_traced(
        &mut self,
        function: BuiltinFunction<C>,
        pc: u64,
        syscall_hash: u32,
    ) {
        let cu_start = self.context_object_pointer.get_remaining();
        let allocation_start = read_syscall_allocation_snapshot();
        let native_start = self.read_syscall_native_instruction_count();
        self.invoke_function(function);
        self.push_syscall_trace(
            pc,
            syscall_hash,
            native_start,
            cu_start,
            allocation_start,
        );
    }

    fn read_syscall_native_instruction_count(&mut self) -> Option<u64> {
        self.syscall_instruction_counter.read()
    }

    fn push_syscall_trace(
        &mut self,
        pc: u64,
        syscall_hash: u32,
        native_start: Option<u64>,
        cu_start: u64,
        allocation_start: Option<SyscallAllocationSnapshot>,
    ) {
        let native_instruction_count = native_start
            .and_then(|native_start| {
                self.read_syscall_native_instruction_count()
                    .map(|end| end.saturating_sub(native_start))
            })
            .unwrap_or(0);
        let compute_units = cu_start.saturating_sub(self.context_object_pointer.get_remaining());
        let allocation_delta = allocation_start
            .and_then(|allocation_start| {
                read_syscall_allocation_snapshot().map(|end| end.delta_since(allocation_start))
            })
            .unwrap_or_default();
        self.syscall_trace.push(SyscallTraceEntry {
            pc,
            syscall_hash,
            compute_units,
            native_instruction_count,
            host_allocations: allocation_delta.allocations,
            host_allocated_bytes: allocation_delta.allocated_bytes,
        });
    }
}

#[cfg(all(feature = "jit", not(target_os = "windows"), target_arch = "x86_64"))]
pub(crate) fn invoke_function_traced_jit<C: ContextObject>(
    vm_addr: *mut EbpfVm<C>,
    function: BuiltinFunction<C>,
    a: u64,
    b: u64,
    c: u64,
    d: u64,
    e: u64,
) {
    let (native_start, cu_start, allocation_start) = unsafe {
        with_vm_from_runtime_environment(vm_addr, |vm| {
            let cu_start = vm.context_object_pointer.get_remaining();
            let allocation_start = read_syscall_allocation_snapshot();
            let native_start = vm.read_syscall_native_instruction_count();
            (native_start, cu_start, allocation_start)
        })
    };
    function(vm_addr, a, b, c, d, e);
    unsafe {
        with_vm_from_runtime_environment(vm_addr, |vm| {
            vm.push_syscall_trace(
                vm.syscall_trace_pc,
                vm.syscall_trace_hash as u32,
                native_start,
                cu_start,
                allocation_start,
            )
        });
    }
}

#[cfg(all(feature = "jit", not(target_os = "windows"), target_arch = "x86_64"))]
unsafe fn with_vm_from_runtime_environment<C: ContextObject, R>(
    vm_addr: *mut EbpfVm<C>,
    f: impl FnOnce(&mut EbpfVm<C>) -> R,
) -> R {
    let vm = unsafe {
        &mut *vm_addr
            .cast::<u64>()
            .offset(-(get_runtime_environment_key() as isize))
            .cast::<EbpfVm<C>>()
    };
    f(vm)
}

#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
struct SyscallInstructionCounter {
    state: SyscallInstructionCounterState,
}

#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
enum SyscallInstructionCounterState {
    Uninitialized,
    Unavailable,
    Ready(PerfInstructionCounter),
}

#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
impl Default for SyscallInstructionCounter {
    fn default() -> Self {
        Self {
            state: SyscallInstructionCounterState::Uninitialized,
        }
    }
}

#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
impl SyscallInstructionCounter {
    fn read(&mut self) -> Option<u64> {
        if matches!(self.state, SyscallInstructionCounterState::Uninitialized) {
            self.state = match PerfInstructionCounter::new() {
                Some(counter) => SyscallInstructionCounterState::Ready(counter),
                None => SyscallInstructionCounterState::Unavailable,
            };
        }
        match &self.state {
            SyscallInstructionCounterState::Ready(counter) => counter.read(),
            SyscallInstructionCounterState::Uninitialized
            | SyscallInstructionCounterState::Unavailable => None,
        }
    }
}

#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
struct PerfInstructionCounter {
    fd: libc::c_int,
    metadata: *const PerfEventMmapPage,
    metadata_size: usize,
}

#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
impl PerfInstructionCounter {
    fn new() -> Option<Self> {
        const PERF_TYPE_HARDWARE: u32 = 0;
        const PERF_COUNT_HW_INSTRUCTIONS: u64 = 1;
        const PERF_FLAG_FD_CLOEXEC: libc::c_ulong = 8;
        const PERF_ATTR_FLAG_EXCLUDE_KERNEL: u64 = 1 << 5;
        const PERF_ATTR_FLAG_EXCLUDE_HV: u64 = 1 << 6;
        const PERF_CAP_USER_RDPMC: u64 = 1 << 2;

        let attr = PerfEventAttr {
            type_: PERF_TYPE_HARDWARE,
            size: std::mem::size_of::<PerfEventAttr>() as u32,
            config: PERF_COUNT_HW_INSTRUCTIONS,
            flags: PERF_ATTR_FLAG_EXCLUDE_KERNEL | PERF_ATTR_FLAG_EXCLUDE_HV,
            ..PerfEventAttr::default()
        };

        let fd = unsafe {
            libc::syscall(
                libc::SYS_perf_event_open,
                &attr as *const PerfEventAttr,
                0i32,
                -1i32,
                -1i32,
                PERF_FLAG_FD_CLOEXEC,
            )
        } as libc::c_int;
        if fd < 0 {
            return None;
        }

        let metadata_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if metadata_size <= 0 {
            unsafe {
                libc::close(fd);
            }
            return None;
        }
        let metadata_size = metadata_size as usize;
        let metadata = unsafe {
            libc::mmap(
                ptr::null_mut(),
                metadata_size,
                libc::PROT_READ,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if metadata == libc::MAP_FAILED {
            unsafe {
                libc::close(fd);
            }
            return None;
        }

        let counter = Self {
            fd,
            metadata: metadata.cast::<PerfEventMmapPage>(),
            metadata_size,
        };
        let capabilities = unsafe { ptr::read_volatile(&(*counter.metadata).capabilities) };
        if capabilities & PERF_CAP_USER_RDPMC == 0 {
            drop(counter);
            return None;
        }
        Some(counter)
    }

    fn read(&self) -> Option<u64> {
        use std::sync::atomic::{compiler_fence, Ordering};

        loop {
            let metadata = unsafe { &*self.metadata };
            let sequence = unsafe { ptr::read_volatile(&metadata.lock) };
            compiler_fence(Ordering::Acquire);
            let index = unsafe { ptr::read_volatile(&metadata.index) };
            let offset = unsafe { ptr::read_volatile(&metadata.offset) };
            let pmc_width = unsafe { ptr::read_volatile(&metadata.pmc_width) };
            if index == 0 {
                return None;
            }
            let pmc = unsafe { rdpmc(index - 1) };
            let pmc = if pmc_width == 0 || pmc_width >= 64 {
                pmc as i64
            } else {
                let shift = 64 - pmc_width as u32;
                ((pmc << shift) as i64) >> shift
            };
            let count = (offset as i128 + pmc as i128) as u64;
            compiler_fence(Ordering::Release);
            let sequence_after = unsafe { ptr::read_volatile(&metadata.lock) };
            if sequence == sequence_after && sequence & 1 == 0 {
                return Some(count);
            }
        }
    }
}

#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
impl Drop for PerfInstructionCounter {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.metadata.cast_mut().cast(), self.metadata_size);
            libc::close(self.fd);
        }
    }
}

#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
#[repr(C)]
struct PerfEventAttr {
    type_: u32,
    size: u32,
    config: u64,
    sample_period: u64,
    sample_type: u64,
    read_format: u64,
    flags: u64,
    wakeup_events: u32,
    bp_type: u32,
    config1: u64,
    config2: u64,
    branch_sample_type: u64,
    sample_regs_user: u64,
    sample_stack_user: u32,
    clockid: i32,
    sample_regs_intr: u64,
    aux_watermark: u32,
    sample_max_stack: u16,
    reserved_2: u16,
    aux_sample_size: u32,
    aux_action: u32,
    sig_data: u64,
    config3: u64,
}

#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
impl Default for PerfEventAttr {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
#[repr(C)]
struct PerfEventMmapPage {
    version: u32,
    compat_version: u32,
    lock: u32,
    index: u32,
    offset: i64,
    time_enabled: u64,
    time_running: u64,
    capabilities: u64,
    pmc_width: u16,
    time_shift: u16,
    time_mult: u32,
    time_offset: u64,
    time_zero: u64,
    size: u32,
    reserved_1: u32,
    time_cycles: u64,
    time_mask: u64,
}

#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
unsafe fn rdpmc(counter: u32) -> u64 {
    let low: u32;
    let high: u32;
    std::arch::asm!(
        "rdpmc",
        in("ecx") counter,
        out("eax") low,
        out("edx") high,
        options(nomem, nostack, preserves_flags),
    );
    ((high as u64) << 32) | low as u64
}

#[cfg(not(all(feature = "jit", target_os = "linux", target_arch = "x86_64")))]
#[derive(Default)]
struct SyscallInstructionCounter;

#[cfg(not(all(feature = "jit", target_os = "linux", target_arch = "x86_64")))]
impl SyscallInstructionCounter {
    fn read(&mut self) -> Option<u64> {
        None
    }
}
