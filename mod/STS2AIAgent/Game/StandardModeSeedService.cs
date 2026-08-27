using System.Reflection;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Multiplayer.Game.Lobby;
using STS2AIAgent.Server;

namespace STS2AIAgent.Game;

internal static class StandardModeSeedService
{
    internal const string Alphabet = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ";
    private static readonly MethodInfo? SeedSetter = ResolveSeedSetter();

    internal static bool IsSupported => SeedSetter != null;

    internal static string CanonicalizeAndValidate(string? input)
    {
        if (input == null)
        {
            throw new ApiException(400, "invalid_request", "set_seed requires game_seed.");
        }

        var canonical = SeedHelper.CanonicalizeSeed(input);
        if (canonical.Length != 10 || canonical.Any(ch => !Alphabet.Contains(ch)))
        {
            throw new ApiException(
                400,
                "invalid_request",
                "game_seed must canonicalize to 10 STS2 seed characters.");
        }

        return canonical;
    }

    internal static string SetAndReadBack(StartRunLobby lobby, string canonical)
    {
        var setter = SeedSetter
            ?? throw new ApiException(
                503,
                "seed_control_unavailable",
                "Exact lobby Seed setter is unavailable.");

        try
        {
            setter.Invoke(lobby, [canonical]);
        }
        catch (TargetInvocationException)
        {
            throw new ApiException(
                503,
                "seed_control_failed",
                "Lobby seed setter invocation failed.");
        }
        catch (Exception exception) when (
            exception is MethodAccessException or ArgumentException or TargetException)
        {
            throw new ApiException(
                503,
                "seed_control_failed",
                "Lobby seed setter invocation failed.");
        }

        if (!StringComparer.Ordinal.Equals(lobby.Seed, canonical))
        {
            throw new ApiException(
                503,
                "seed_verification_failed",
                "Lobby seed read-back did not match the requested seed.");
        }

        return lobby.Seed;
    }

    private static MethodInfo? ResolveSeedSetter()
    {
        var property = typeof(StartRunLobby).GetProperty(
            "Seed",
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
        if (property == null || property.PropertyType != typeof(string))
        {
            return null;
        }

        var setter = property.GetSetMethod(nonPublic: true);
        if (setter == null ||
            setter.IsPublic ||
            setter.IsStatic ||
            setter.ReturnType != typeof(void) ||
            setter.DeclaringType != typeof(StartRunLobby))
        {
            return null;
        }

        var parameters = setter.GetParameters();
        return parameters.Length == 1 && parameters[0].ParameterType == typeof(string)
            ? setter
            : null;
    }
}
