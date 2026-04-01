"""RePaint resampling schedule."""


def get_schedule_jump(t_T, n_sample, jump_length, jump_n_sample,
                      jump2_length=1, jump2_n_sample=1,
                      jump3_length=1, jump3_n_sample=1,
                      start_resampling=100000000):
    jumps = {}
    for j in range(0, t_T - jump_length, jump_length):
        jumps[j] = jump_n_sample - 1

    jumps2 = {}
    for j in range(0, t_T - jump2_length, jump2_length):
        jumps2[j] = jump2_n_sample - 1

    jumps3 = {}
    for j in range(0, t_T - jump3_length, jump3_length):
        jumps3[j] = jump3_n_sample - 1

    t = t_T
    ts = []

    while t >= 1:
        t = t - 1
        ts.append(t)

        if t + 1 < t_T - 1 and t <= start_resampling:
            for _ in range(n_sample - 1):
                t = t + 1
                ts.append(t)
                if t >= 0:
                    t = t - 1
                    ts.append(t)

        if jumps3.get(t, 0) > 0 and t <= start_resampling - jump3_length:
            jumps3[t] = jumps3[t] - 1
            for _ in range(jump3_length):
                t = t + 1
                ts.append(t)

        if jumps2.get(t, 0) > 0 and t <= start_resampling - jump2_length:
            jumps2[t] = jumps2[t] - 1
            for _ in range(jump2_length):
                t = t + 1
                ts.append(t)
            jumps3 = {}
            for j in range(0, t_T - jump3_length, jump3_length):
                jumps3[j] = jump3_n_sample - 1

        if jumps.get(t, 0) > 0 and t <= start_resampling - jump_length:
            jumps[t] = jumps[t] - 1
            for _ in range(jump_length):
                t = t + 1
                ts.append(t)
            jumps2 = {}
            for j in range(0, t_T - jump2_length, jump2_length):
                jumps2[j] = jump2_n_sample - 1
            jumps3 = {}
            for j in range(0, t_T - jump3_length, jump3_length):
                jumps3[j] = jump3_n_sample - 1

    ts.append(-1)

    # Validate
    assert ts[0] > ts[1]
    assert ts[-1] == -1
    for t_last, t_cur in zip(ts[:-1], ts[1:]):
        assert abs(t_last - t_cur) == 1

    return ts
