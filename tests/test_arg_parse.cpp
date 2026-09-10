#include <cassert>
#include <cstdio>
#include "arg_parse.hpp"

using namespace az;

void test_accepts_positive_integers() {
    assert(parsePositiveIntArg("1") == 1);
    assert(parsePositiveIntArg("200") == 200);
    assert(parsePositiveIntArg("2147483647") == 2147483647);
    assert(parsePositiveIntArg("+7") == 7);
}

void test_rejects_non_numeric_and_trailing_garbage() {
    assert(!parsePositiveIntArg(nullptr));
    assert(!parsePositiveIntArg(""));
    assert(!parsePositiveIntArg("abc"));
    assert(!parsePositiveIntArg("20x"));
    assert(!parsePositiveIntArg("1 2"));
    assert(!parsePositiveIntArg("1.5"));
}

void test_rejects_zero_negative_and_overflow() {
    // std::atoi returns 0 for all of these, which silently became "run no
    // iterations at all" before the counts were validated.
    assert(!parsePositiveIntArg("0"));
    assert(!parsePositiveIntArg("-1"));
    assert(!parsePositiveIntArg("2147483648"));
    assert(!parsePositiveIntArg("99999999999999999999"));
}

int main() {
    test_accepts_positive_integers();
    test_rejects_non_numeric_and_trailing_garbage();
    test_rejects_zero_negative_and_overflow();
    std::printf("all arg_parse tests passed\n");
    return 0;
}
