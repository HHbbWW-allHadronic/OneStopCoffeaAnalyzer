from textual.app import App, ComposeResult
from textual import on
from textual.reactive import reactive
from textual.widgets import (
    Header,
    Tree,
    Footer,
    Static,
    Pretty,
    Input,
    Select,
    Label,
    RichLog,
)
from textual.containers import Horizontal, Vertical, VerticalScroll
from analyzer.core.results import ResultGroup, Histogram
from textual.widget import Widget
import numpy as np
import hist


def formatSize(size_in_bytes: int) -> str:
    i = ("B", "KB", "MB", "GB")
    for unit in i:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.1f} {unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.1f} TB"


class ResultBrowser(App):
    CSS = """
    #sidebar {
        width: 30%;
        height: 100%;
        dock: left;
    }
    #search_input {
        dock: top;
        margin: 1;
    }
    ResultTree {
        width: 100%;
        height: 100%;
    }
    ResultViewer {
        width: 70%;
        height: 100%;
        padding-left: 2;
        border-left: solid $panel;
    }
    HistogramViewer {
        width: 100%;
        height: 100%;
        padding: 1;
    }
    .axis_controls {
        height: auto;
        min-height: 5;
        margin-bottom: 1;
    }
    .axis_control {
        margin: 1;
        padding: 0;
        width: 1fr;
        height: auto;
    }
    #hist_control_grid {
        height: auto;
        min-height: 5;
    }
    PlotextPlot {
        height: 1fr;
        min-height: 20;
    }
    #console_container {
        height: 15;
        border-top: solid $panel;
        dock: bottom;
    }
    #console_log {
        height: 1fr;
    }
    #console_input {
        dock: bottom;
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
                with Vertical(id="sidebar"):
                    yield Input(placeholder="Search results...", id="search_input")
                    yield ResultTree(self.results, id="result_tree")
                yield self.viewer
            with Vertical(id="console_container"):
                yield RichLog(id="console_log", highlight=True, markup=True)
                yield Input(
                    placeholder="Interactive Console (available: RESULT)",
                    id="console_input",
                )
            yield Footer()

    @on(Input.Submitted, "#console_input")
    def handleConsoleSubmit(self, event):
        input_widget = event.control
        log_widget = self.query_one("#console_log", RichLog)

        cmd = event.value.strip()
        if not cmd:
            return

        input_widget.value = ""
        log_widget.write(f">>> {cmd}")

        current_result = getattr(self.viewer, "result", None)

        local_vars = {"RESULT": current_result}
        try:
            try:
                out = eval(cmd, globals(), local_vars)
                if out is not None:
                    log_widget.write(repr(out))
            except SyntaxError:
                import io
                import sys
                from contextlib import redirect_stdout

                f = io.StringIO()
                with redirect_stdout(f):
                    exec(cmd, globals(), local_vars)
                output = f.getvalue()
                if output:
                    log_widget.write(output)
        except Exception as e:
            log_widget.write(f"[red]{type(e).__name__}: {str(e)}[/red]")

    @on(Input.Changed, "#search_input")
    def handleSearch(self, event):
        tree = self.query_one("#result_tree", ResultTree)
        tree.search_term = event.value.lower()

    @on(Tree.NodeSelected)
    def handleSelected(self, message):
        self.viewer.showResult(message.node.data)

    def on_button_pressed(self) -> None:
        self.exit()


class ResultTree(Widget):
    search_term = reactive("", recompose=True)

    def __init__(self, results, **kwargs):
        super().__init__(**kwargs)
        self.results = results

    def compose(self) -> ComposeResult:
        tree = Tree("/")

        def matches(result, term):
            if term in result.name.lower():
                return True
            if isinstance(result, ResultGroup):
                for child_name in result:
                    if matches(result[child_name], term):
                        return True
            return False

        def handleResult(parent, result, term):
            if term and not matches(result, term):
                return

            size_str = formatSize(result.approxSize())
            label = f"{result.name} ({size_str})"

            if isinstance(result, ResultGroup):
                new_node = parent.add(label, data=result)
                if term:
                    new_node.expand()
                for child_name in result:
                    handleResult(new_node, result[child_name], term)
                return new_node
            else:
                parent.add_leaf(label, data=result)

        handleResult(tree.root, self.results, self.search_term)
        tree.root.expand()
        yield tree


class HistogramViewer(Widget):
    def __init__(self, histogram_result: Histogram, **kwargs):
        super().__init__(**kwargs)
        self.histogram_result = histogram_result
        self.hist_obj = histogram_result.histogram
        self.axes = self.hist_obj.axes

    def getAxisOptions(self, axis):
        options = [("X-Axis", "x"), ("Y-Axis", "y"), ("Project (Sum)", "sum")]
        for idx in range(len(axis)):
            if type(axis).__name__ == "StrCategory":
                val = axis[idx]
                options.append((f"Select: {val}", f"sel_{idx}"))
            elif hasattr(axis, "centers"):
                options.append((f"Select: {axis.centers[idx]:.4g}", f"sel_{idx}"))
            else:
                options.append((f"Select: Bin {idx}", f"sel_{idx}"))
        return options

    def compose(self) -> ComposeResult:
        from textual.containers import Vertical, Horizontal
        from textual_plotext import PlotextPlot

        with Vertical():
            with Horizontal(classes="axis_controls", id="hist_control_grid"):
                for i, axis in enumerate(self.axes):
                    with Vertical(classes="axis_control"):
                        yield Label(axis.name)

                        default_val = "sum"
                        if i == len(self.axes) - 1:
                            default_val = "x"
                        elif axis.name == "variation":
                            for idx in range(len(axis)):
                                if axis[idx] == "central":
                                    default_val = f"sel_{idx}"
                                    break

                        sel = Select(
                            self.getAxisOptions(axis),
                            value=default_val,
                            id=f"axis_select_{i}",
                        )
                        yield sel
            yield PlotextPlot(id="hist_plot")

    def on_mount(self):
        self.set_timer(0.1, self.updatePlot)

    @on(Select.Changed)
    def updatePlot(self, event=None):
        from textual_plotext import PlotextPlot

        try:
            plot_widget = self.query_one("#hist_plot", PlotextPlot)
            plt = plot_widget.plt
            plt.clear_figure()
            plt.theme("dark")

            selections = []
            for i in range(len(self.axes)):
                sel = self.query_one(f"#axis_select_{i}", Select).value
                selections.append(sel)

            slice_dict = {}
            target_x_name, target_y_name = None, None

            for i, (axis, sel) in enumerate(zip(self.axes, selections)):
                if sel == "x":
                    target_x_name = axis.name
                elif sel == "y":
                    target_y_name = axis.name
                elif sel == "sum":
                    slice_dict[axis.name] = sum
                elif str(sel).startswith("sel_"):
                    idx = int(str(sel).split("_")[1])
                    slice_dict[axis.name] = idx

            if not target_x_name:
                plt.title("No X-Axis selected")
                plot_widget.refresh()
                return

            sliced_h = self.hist_obj[slice_dict]

            if target_y_name:
                axes_names = list(sliced_h.axes.name)
                try:
                    x_idx = axes_names.index(target_x_name)
                    y_idx = axes_names.index(target_y_name)
                    x_axis = sliced_h.axes[x_idx]
                    y_axis = sliced_h.axes[y_idx]

                    vals = sliced_h.values()
                    if x_idx != 1:
                        vals = vals.T

                    x_label = (
                        x_axis.label if hasattr(x_axis, "label") else target_x_name
                    )
                    y_label = (
                        y_axis.label if hasattr(y_axis, "label") else target_y_name
                    )

                    plt.title(f"{y_label} vs {x_label}")
                    plt.xlabel(x_label)
                    plt.ylabel(y_label)

                    plot_data = np.flipud(vals).tolist()
                    if hasattr(plt, "matrix_plot"):
                        plt.matrix_plot(plot_data)

                        x_centers = (
                            x_axis.centers
                            if hasattr(x_axis, "centers")
                            else list(x_axis)
                        )
                        y_centers = (
                            y_axis.centers
                            if hasattr(y_axis, "centers")
                            else list(y_axis)
                        )
                        try:
                            if hasattr(plt, "xticks"):
                                plt.xticks(
                                    list(range(len(x_centers))),
                                    [
                                        f"{x:.2f}" if isinstance(x, float) else str(x)
                                        for x in x_centers
                                    ],
                                )
                            if hasattr(plt, "yticks"):
                                plt.yticks(
                                    list(range(len(y_centers))),
                                    [
                                        f"{y:.2f}" if isinstance(y, float) else str(y)
                                        for y in reversed(y_centers)
                                    ],
                                )
                        except Exception:
                            pass
                    else:
                        plt.heatmap(plot_data)
                except Exception as e:
                    plt.title(f"2D plotting error: {e}")
            else:
                x_axis = sliced_h.axes[0]
                x_label = x_axis.label if hasattr(x_axis, "label") else target_x_name

                vals = sliced_h.values()
                centers = x_axis.centers if hasattr(x_axis, "centers") else list(x_axis)

                plt.title(x_label)
                plt.xlabel(x_label)
                plt.ylabel("Events")
                try:
                    plt.bar(list(centers), list(vals))
                except Exception:
                    plt.plot(list(centers), list(vals))

            plot_widget.refresh()
        except Exception as e:
            try:
                plot_widget = self.query_one("#hist_plot", PlotextPlot)
                plot_widget.plt.clear_figure()
                plot_widget.plt.title(f"Plotting Error: {e}")
                plot_widget.refresh()
            except Exception:
                pass


class ResultViewer(Widget):
    result = reactive(None, recompose=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def showResult(self, result):
        self.result = result

    def compose(self):
        if self.result is None:
            return

        if isinstance(self.result, Histogram):
            yield HistogramViewer(self.result)
            return

        w = self.result.widget()
        if w is None:
            with VerticalScroll():
                try:
                    rep = repr(self.result)
                    if len(rep) > 10000:
                        yield Label(
                            f"Result formatted output is too long to display ({len(rep)} characters)."
                        )
                    else:
                        yield Pretty(self.result)
                except Exception as e:
                    yield Label(f"Could not display result: {e}")
        else:
            yield w
