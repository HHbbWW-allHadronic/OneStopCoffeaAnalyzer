def getPremadeExcutors():
    from analyzer.core.executors.immediate_exec import ImmediateExecutor

    return {"imm_1000_local": ImmediateExecutor(chunk_size=1000)}
