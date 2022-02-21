import numpy as np


def brute_force(n, D):
    """
    returns how many n digit numbers have a digit sum = 0 (mod D).
    """

    def inner(num):
        if len(num) == n:
            if sum([int(digit) for digit in num]) % D == 0:
                return 1
            return 0

        result = 0
        for digit in '0123456789':
            result += inner(num + digit)

        return result

    return inner('')


def first_dp(n, D):
    """
    returns how many n digit numbers have a digit sum = 0 (mod D).
    Uses the first DP algorithm discussed in the video.
    """

    dp = {}
    for digit in range(10):
        state = (1, digit % D)
        if state in dp:
            dp[state] += 1
        else:
            dp[state] = 1

    for i in range(2, n + 1):
        for k in range(D):
            state = (i, k)
            dp[state] = 0
            for digit in range(10):
                # note: negative numbers % D might be handled
                # differently by other programming languages.
                # In Python, the result will always be positive.
                # For example, -3 % 5 is 2 in Python, but -3 would also
                # be mathematically correct.
                # How could we make it positive regardless of how the language
                # handles it?

                prev_state = (i - 1, (k - digit) % D)
                if prev_state in dp:
                    dp[state] += dp[prev_state]

    return dp[(n, 0)]


def second_dp(n, D):
    """
    returns how many n digit numbers have a digit sum = 0 (mod D).
    Uses the second DP algorithm discussed in the video.
    """

    # let's use a proper array this time
    dp = np.zeros((int(np.log2(n)) + 1, D), dtype='object') # also try np.int64
    for digit in range(10):
        dp[0, digit % D] += 1

    k = 1
    while 2**k <= n: # exercise: how can we optimize this?
        for s1 in range(D):
            for s2 in range(D):
                dp[k, (s1 + s2) % D] += dp[k - 1, s1] * dp[k - 1, s2]
        k += 1
    k -= 1

    if 2**k == n:
        return dp[k, 0]

    # exercise: do we need all the log2(n) lines?
    dp2 = np.zeros((k + 1, D), dtype='object') # also try np.int64

    # exercise: break this down and figure it out from the inside out
    binary_rep = bin(n)[::-1][:-2]
    # we start from the least significant bit and build up the answer
    p1 = binary_rep.find('1')
    i = 0
    dp2[i, :] = dp[p1, :]
    i += 1

    for pi, bit in enumerate(binary_rep):
        if pi > p1 and bit == '1':
            for s1 in range(D):
                for s2 in range(D):
                    dp2[i, (s1 + s2) % D] += dp2[i - 1, s1] * dp[pi, s2]
            i += 1
    return dp2[i-1, 0]


def run_tests_battery():
    # can only compare so much with the brute force function
    for n in range(1, 5):
        for D in range(1, 30):
            brute_res = brute_force(n, D)
            first_dp_res = first_dp(n, D)
            assert brute_res == first_dp_res, f'brute({n}, {D})={brute_res}, first_dp({n}, {D})={first_dp_res}'

    # can compare a lot more between the DPs
    for n in range(1, 50):
        for D in range(1, 30):
            first_dp_res = first_dp(n, D)
            second_dp_res = second_dp(n, D)
            assert first_dp_res == second_dp_res, f'first_dp({n}, {D})={first_dp_res}, second_dp({n}, {D})={second_dp_res}'


run_tests_battery()

# Note: this is still slow because we use big integers instead of the more efficient
# np.int64. Try computing the results modulo something and see if that help things
# to be faster.

# Implementing this in C will also help :).
#print(second_dp(10000, 97))
#print(first_dp(10000, 97))
