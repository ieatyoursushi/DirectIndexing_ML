using System.Diagnostics;

namespace DirectIndexing.ML;

/// <summary>
/// Minimal bridge to the Python ML pipeline.
///
/// Shells out to <c>uv run python -m &lt;module&gt; [args...]</c> from
/// <c>src/ML/Python/</c>, streams stdout/stderr to the host console, and returns
/// the exit code.
///
/// No data passing in memory — Python writes files to <c>data/artifacts/</c> and
/// <c>src/Export/{eda,models}/</c>, C# consumes them on subsequent runs (today:
/// nothing consumes them yet; v0.3 will read <c>cluster_assignments.json</c> at
/// harvest time).
/// </summary>
public static class PythonRunner
{
    /// <summary>
    /// Run a Python module via uv. Returns the process exit code; non-zero indicates failure.
    /// </summary>
    /// <param name="scriptModule">Dotted module path, e.g. "scripts.run_eda"</param>
    /// <param name="args">Extra args passed to the script</param>
    public static int Run(string scriptModule, params string[] args)
    {
        // Locate the repo root by walking upward from CWD until we find DirectIndexing.sln.
        // Robust to whether `dotnet run` was invoked from the repo root or from src/.
        string? root = Directory.GetCurrentDirectory();
        while (root != null && !File.Exists(Path.Combine(root, "DirectIndexing.sln")))
            root = Path.GetDirectoryName(root);

        if (root == null)
        {
            Console.Error.WriteLine("[PythonRunner] Could not locate repo root " +
                                    "(no ancestor of CWD contains DirectIndexing.sln).");
            return 127;
        }

        string mlDir = Path.Combine(root, "src", "ML", "Python");
        if (!Directory.Exists(mlDir))
        {
            Console.Error.WriteLine($"[PythonRunner] Python pipeline not found at {mlDir}.");
            return 127;
        }

        var argLine = $"run python -m {scriptModule} {string.Join(' ', args)}";
        Console.WriteLine($"[PythonRunner] cwd={mlDir}");
        Console.WriteLine($"[PythonRunner] uv {argLine}");

        var psi = new ProcessStartInfo("uv", argLine)
        {
            WorkingDirectory       = mlDir,
            RedirectStandardOutput = true,
            RedirectStandardError  = true,
            UseShellExecute        = false,
        };

        try
        {
            using var p = Process.Start(psi)!;
            p.OutputDataReceived += (_, e) => { if (e.Data != null) Console.WriteLine(e.Data); };
            p.ErrorDataReceived  += (_, e) => { if (e.Data != null) Console.Error.WriteLine(e.Data); };
            p.BeginOutputReadLine();
            p.BeginErrorReadLine();
            p.WaitForExit();
            return p.ExitCode;
        }
        catch (System.ComponentModel.Win32Exception ex)
        {
            Console.Error.WriteLine($"[PythonRunner] Failed to launch `uv`. " +
                                    $"Install via `brew install uv`. Error: {ex.Message}");
            return 127;
        }
    }
}
