def jumpIn(**kwargs):
    import code
    import readline
    import rlcompleter

    vars = globals()
    vars.update(locals())
    vars.update(kwargs)
    readline.set_completer(rlcompleter.Completer(vars).complete)
    readline.parse_and_bind("tab: complete")
    code.InteractiveConsole(vars).interact()
