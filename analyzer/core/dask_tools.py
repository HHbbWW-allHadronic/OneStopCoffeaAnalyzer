import math

def reduceResults(
    client,
    reduction_function,
    futures,
    reduction_factor=5,
    target_final_count=10,
    close_to_target_frac=0.7,
    key_suffix="",
):

    layer = 0
    while len(futures) > target_final_count:
        if (len(futures) / reduction_factor) < (
            target_final_count * close_to_target_frac
        ):
            reduction_factor = math.ceil(len(futures) / target_final_count)
        futures = [
            client.submit(
                reduction_function,
                futures[i : i + reduction_factor],
                key=f"merge-{layer}-{i}" + str(key_suffix),
            )
            for i in range(0, len(futures), reduction_factor)
        ]

        layer += 1
    return futures
