#pragma once

#include <windows.h>

// The FAULTING thread's return addresses, walked from the exception's own context (x64 unwind data).
// CaptureStackBackTrace from inside a vectored handler walks the HANDLER's stack and stops at the dispatcher when the
// faulting frame has no unwind data -- a call through a null pointer is exactly that: in Cyberpunk 2077
// "execute of 0" logged a trace of the reporter and three ntdll frames, nothing of who called. Here a frame without
// unwind data (address 0, a leaf) is left by its return address on top of the stack. Reads stay inside this thread's
// stack. Offline test: tools/xr_observer/test/crash_trace_test.
inline int WalkFaultStack(const CONTEXT* fault, void** frames, int max)
{
    if (fault == nullptr || frames == nullptr || max <= 0)
        return 0;
    ULONG_PTR low = 0, high = 0;
    GetCurrentThreadStackLimits(&low, &high);
    CONTEXT ctx = *fault;
    int n = 0;
    for (int guard = 0; n < max && guard < max + 4; ++guard)
    {
        frames[n++] = (void*) ctx.Rip;
        DWORD64 imageBase = 0;
        PRUNTIME_FUNCTION fn = ctx.Rip != 0 ? RtlLookupFunctionEntry(ctx.Rip, &imageBase, nullptr) : nullptr;
        if (fn == nullptr)
        {
            if (ctx.Rsp < low || ctx.Rsp + 8 > high)
                break;
            ctx.Rip = *(const DWORD64*) ctx.Rsp;
            ctx.Rsp += 8;
        }
        else
        {
            void* handlerData = nullptr;
            DWORD64 establisher = 0;
            RtlVirtualUnwind(UNW_FLAG_NHANDLER, imageBase, ctx.Rip, fn, &ctx, &handlerData, &establisher, nullptr);
        }
        if (ctx.Rip == 0 || ctx.Rsp < low || ctx.Rsp > high)
            break;
    }
    return n;
}
