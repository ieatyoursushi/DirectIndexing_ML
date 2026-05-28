using System.Diagnostics;

namespace DirectIndexing.ML;

/// <summary>
/// Subprocess wrapper for invoking the thin Python renderer (EDA + report).
/// All ML logic lives in C#; Python only does plotting and HTML stitching
/// — the architecture's clean Python ↔ C# seam.
///
/// Runs <c>uv run python -m {scriptModule} {args}</c> from
/// <c>src/ML/Python/</c>, streaming stdout/stderr to this process's console.
/// </summary>
public static class PythonRunner
{
    public static int Run(string scriptModule, params string[] args)
    {
        string root = LocateRepoRoot();
        string mlDir = Path.Combine(root, "src", "ML", "Python");

        var psi = new ProcessStartInfo("uv", $"run python -m {scriptModule} {string.Join(' ', args)}")
        {
            WorkingDirectory = mlDir,
            RedirectStandardOutput = true,
            RedirectStandardError  = true,
            UseShellExecute = false,
            CreateNoWindow  = true,
        };

        Console.WriteLine($"[PythonRunner] cwd={mlDir}  cmd=uv run python -m {scriptModule} {string.Join(' ', args)}");

        using var p = Process.Start(psi)
            ?? throw new InvalidOperationException("failed to start uv subprocess");

        p.OutputDataReceived += (_, e) => { if (e.Data != null) Console.WriteLine(e.Data); };
        p.ErrorDataReceived  += (_, e) => { if (e.Data != null) Console.Error.WriteLine(e.Data); };
        p.BeginOutputReadLine();
        p.BeginErrorReadLine();
        p.WaitForExit();
        return p.ExitCode;
    }

    private static string LocateRepoRoot()
    {
        var dir = new DirectoryInfo(Directory.GetCurrentDirectory());
        while (dir != null)
        {
            if (File.Exists(Path.Combine(dir.FullName, "DirectIndexing.sln")))
                return dir.FullName;
            dir = dir.Parent;
        }
        throw new InvalidOperationException(
            $"could not find DirectIndexing.sln from {Directory.GetCurrentDirectory()}");
    }
}
