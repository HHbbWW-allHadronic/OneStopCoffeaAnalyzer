#from analyzer.cli import main
#
#if __name__ == "__main__":
#    main()
#
#

from analyzer.core import MODULE_REPO

if __name__ == '__main__':
    from rich import print
    import analyzer.modules 
    print(MODULE_REPO)
    print(id(MODULE_REPO))
