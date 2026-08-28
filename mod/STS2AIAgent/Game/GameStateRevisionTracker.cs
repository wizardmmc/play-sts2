using System.Text.Json;

namespace STS2AIAgent.Game;

/// <summary>
/// 为智能体可见状态分配单调 revision。revision 只在完整状态内容变化时推进，
/// 动作端点可据此在游戏线程内拒绝基于旧观测生成的动作。
/// </summary>
internal static class GameStateRevisionTracker
{
    private static readonly object Gate = new();
    private static readonly JsonSerializerOptions FingerprintOptions = new()
    {
        PropertyNamingPolicy = null,
        WriteIndented = false
    };

    private static long _revision;
    private static string? _snapshotJson;

    public static GameStatePayload Stamp(GameStatePayload state)
    {
        lock (Gate)
        {
            state.state_revision = 0;
            var json = JsonSerializer.Serialize(state, FingerprintOptions);
            if (!string.Equals(
                    json,
                    _snapshotJson,
                    StringComparison.Ordinal))
            {
                _snapshotJson = json;
                _revision++;
            }

            state.state_revision = _revision;
            return state;
        }
    }
}
