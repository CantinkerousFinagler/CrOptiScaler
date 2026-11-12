# nvngx.dll_dlssnr.dll

A small module that forwards OptiScaler's Neural Rendering calls (create, evaluate, release) to the
model, `nvngx_dlssnr.dll`. It contains no NVIDIA code.

The prebuilt DLL is committed because it is 12 KB, changes almost never, and has to be in the package
for a drop-in build to work at all. To rebuild it:

    cmake -S . -B build -A x64
    cmake --build build --config Release

One detail is load-bearing and not obvious. The forwarder must not `return modelFn(...)`: a tail call
would leave this module's frame behind, and the call has to return through this module. The result
goes into a `volatile` local first.
