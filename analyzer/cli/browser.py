from textual.app import App, ComposeResult
from textual import on
from rich.console import Console
from textual.widgets import Welcome, Header, Tree, Footer, Static, Pretty
from textual.containers import Horizontal, Vertical
from analyzer.core.results import ResultGroup
from textual.widget import Widget


class ResultBrowser(App):
    CSS = """
    ResultTree{
    width: 1fr;
    }
    ResultViewer{
    width: 2fr;
    }

    """
    TITLE = "Result Browser"

    def __init__(self, results):
        super().__init__()
        self.results = results
        self.viewer = ResultViewer()

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Header()
            with Horizontal():
                yield ResultTree(self.results)
                yield self.viewer
            yield Footer()

    @on(Tree.NodeSelected)
    def handleSelected(self, message):
        self.viewer.showResult(message.node.data)

    def on_button_pressed(self) -> None:
        self.exit()


class ResultTree(Static):

    def __init__(self, results, **kwargs):
        super().__init__(**kwargs)
        self.results = results

    def compose(self) -> ComposeResult:
        tree = Tree("/")
        tree.root.expand()

        def handleResult(parent, result):
            if isinstance(result, ResultGroup):
                new_node = parent.add(result.name, data=result)
                new_node.expand()
                for child_name in result:
                    handleResult(new_node, result[child_name])
                return new_node
            else:
                parent.add_leaf(result.name, data=result)

        handleResult(tree.root, self.results)
        yield tree


class ResultViewer(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.result = None

    def showResult(self, result):
        self.result = result

    def compose(self):
        if self.result is None:
            return
        w = self.result.renderWidget()
        if w is None:
            yield Pretty(result)
        else:
            yield w
