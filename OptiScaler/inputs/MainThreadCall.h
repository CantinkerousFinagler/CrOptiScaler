#pragma once

// Runs a callback on a window's thread, in message context, from any thread of the process (first use:
// a game's engine resize, called where the game's own WM_SIZE handler would call it).
//
// The window is subclassed once (SetWindowLongPtr GWLP_WNDPROC; the process's own window, so any of its threads may
// do it); every message but ours goes to the procedure that was there. Ours is a registered message: on it the
// requests posted so far are run, on the window's thread, between two of the game's frames. Nothing is undone at
// exit: the DLL lives as long as the game.
//
// Header only, no log: the caller reads the counts. tools/xr_observer/test/overlay_test "mainthread" runs it on a
// real window of another thread.

#include <windows.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

namespace MainThreadCall
{
using Callback = void (*)(void* context);

class Runner
{
  public:
    static Runner& Get()
    {
        static Runner runner;
        return runner;
    }

    // Subclasses `window` (once; a second call with the same window is a yes, with another window a no). FALSE with
    // `why` set when it cannot be done.
    bool Attach(HWND window, const char** why)
    {
        std::lock_guard lock(_mutex);
        if (_window != nullptr)
        {
            if (_window == window)
                return true;
            if (why != nullptr)
                *why = "another window is attached already";
            return false;
        }
        if (window == nullptr || !IsWindow(window))
        {
            if (why != nullptr)
                *why = "no window";
            return false;
        }
        if (_message == 0)
            _message = RegisterWindowMessageW(L"OptiScaler.MainThreadCall");
        if (_message == 0)
        {
            if (why != nullptr)
                *why = "no message could be registered";
            return false;
        }
        SetLastError(0);
        const LONG_PTR before = SetWindowLongPtrW(window, GWLP_WNDPROC, (LONG_PTR) &Runner::Proc);
        if (before == 0)
        {
            if (why != nullptr)
                *why = "SetWindowLongPtr refused";
            return false;
        }
        _original = (WNDPROC) before;
        _window = window;
        _windowThread = GetWindowThreadProcessId(window, nullptr);
        return true;
    }

    // Queues `callback(context)` and wakes the window's thread. FALSE with `why` set when nothing was posted.
    bool Post(Callback callback, void* context, const char** why)
    {
        HWND window = nullptr;
        UINT message = 0;
        {
            std::lock_guard lock(_mutex);
            if (_window == nullptr)
            {
                if (why != nullptr)
                    *why = "no window is attached";
                return false;
            }
            _queue.push_back({ callback, context });
            window = _window;
            message = _message;
        }
        if (!PostMessageW(window, message, 0, 0))
        {
            std::lock_guard lock(_mutex);
            if (!_queue.empty())
                _queue.pop_back();
            if (why != nullptr)
                *why = "PostMessage refused";
            return false;
        }
        ++_posted;
        return true;
    }

    HWND Window() const { return _window; }
    DWORD WindowThread() const { return _windowThread; }
    uint64_t Posted() const { return _posted.load(); }
    uint64_t Ran() const { return _ran.load(); }
    DWORD LastRanOn() const { return _lastThread.load(); } // the thread the last callback ran on
    uint64_t Passed() const { return _passed.load(); }    // messages handed to the procedure that was there

  private:
    struct Request
    {
        Callback callback;
        void* context;
    };

    static LRESULT CALLBACK Proc(HWND window, UINT message, WPARAM wParam, LPARAM lParam)
    {
        Runner& self = Get();
        if (message != 0 && message == self._message)
        {
            std::vector<Request> requests;
            {
                std::lock_guard lock(self._mutex);
                requests.swap(self._queue);
            }
            for (const Request& request : requests)
            {
                if (request.callback != nullptr)
                    request.callback(request.context);
                ++self._ran;
            }
            self._lastThread = GetCurrentThreadId();
            return 0;
        }
        ++self._passed;
        return CallWindowProcW(self._original, window, message, wParam, lParam);
    }

    std::mutex _mutex;
    HWND _window = nullptr;
    WNDPROC _original = nullptr;
    UINT _message = 0;
    DWORD _windowThread = 0;
    std::vector<Request> _queue;
    std::atomic<uint64_t> _posted { 0 }, _ran { 0 }, _passed { 0 };
    std::atomic<DWORD> _lastThread { 0 };
};
} // namespace MainThreadCall
