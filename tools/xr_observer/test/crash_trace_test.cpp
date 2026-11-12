// Offline test of WalkFaultStack (OptiScaler/inputs/CrashTrace.h), the crash reporter's trace of the FAULTING thread.
// A function of ours calls through a null pointer (an "execute of 0"); a vectored handler,
// as the reporter is one, walks the stack and must find that caller and the caller's caller.
//   crash_trace_test
// Control, must FAIL:
//   CRASH_TEST_OLD=1  the trace as it was (CaptureStackBackTrace from the handler): the caller is not in it
#include <windows.h>

#include "../../../OptiScaler/inputs/CrashTrace.h"

#include <cstdio>

namespace
{
bool g_old = false;
using Fn = void (*)();
volatile Fn g_null = nullptr;
volatile int g_sink = 0;

__declspec(noinline) void Caller()
{
    g_sink = 1;
    g_null(); // execute of 0
    g_sink = 2;
}

__declspec(noinline) void Outer()
{
    g_sink = 3;
    Caller();
    g_sink = 4;
}

// A return address just after the call in these tiny functions (measured +0x13..+0x17); 0x40, not more: the handler
// lies right after Caller in the image and must not count as it.
bool Inside(void* pc, void* fn) { return (char*) pc > (char*) fn && (char*) pc < (char*) fn + 0x40; }

LONG CALLBACK Handler(EXCEPTION_POINTERS* ep)
{
    void* frames[24] {};
    const int n = g_old ? (int) CaptureStackBackTrace(0, 24, frames, nullptr)
                        : WalkFaultStack(ep->ContextRecord, frames, 24);
    bool caller = false, outer = false;
    std::printf("%s, %d frames:", g_old ? "the OLD trace (the handler's own stack)" : "the faulting thread's stack", n);
    for (int i = 0; i < n; ++i)
    {
        std::printf(" %p", frames[i]);
        caller = caller || Inside(frames[i], (void*) &Caller);
        outer = outer || Inside(frames[i], (void*) &Outer);
    }
    std::printf("\nthe faulting address %p; Caller at %p %s; Outer at %p %s\n", ep->ExceptionRecord->ExceptionAddress,
                (void*) &Caller, caller ? "FOUND" : "missing", (void*) &Outer, outer ? "FOUND" : "missing");
    const bool ok = n > 0 && frames[0] == nullptr && caller && outer;
    std::printf("%s: 1 checks, %d failures\n", ok ? "PASSED" : "FAILED", ok ? 0 : 1);
    std::fflush(stdout);
    ExitProcess(ok ? 0 : 1);
}
} // namespace

int main()
{
    char value[8] = {};
    g_old = GetEnvironmentVariableA("CRASH_TEST_OLD", value, sizeof(value)) > 0 && value[0] == '1';
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    AddVectoredExceptionHandler(1, Handler);
    Outer();
    std::printf("FAILED: no exception\n");
    return 1;
}
