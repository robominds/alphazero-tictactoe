#pragma once
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <optional>

namespace az {

// Parses a strictly positive decimal integer from a command-line argument,
// returning nullopt for anything else: empty strings, non-numeric text,
// trailing garbage ("20x"), zero, negatives, and values above INT_MAX.
//
// std::atoi cannot report any of these -- it returns 0 both for "0" and for
// "abc", and its behavior on overflow is undefined -- so a typo silently
// turns into a run that does no iterations at all. Every count read from
// argv goes through here so the caller can fail loudly instead.
inline std::optional<int> parsePositiveIntArg(const char* text) {
    if (text == nullptr || *text == '\0') return std::nullopt;
    errno = 0;
    char* end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') return std::nullopt;
    if (value < 1 || value > std::numeric_limits<int>::max()) return std::nullopt;
    return static_cast<int>(value);
}

} // namespace az
