import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def addCmsInfo(ax, pos="in", additional_text=None, color="black") -> mpl.axis.Axis:
    text = r"$\bf{CMS}\ \it{Preliminary}$"
    
    if additional_text:
        text += additional_text
    if pos == "in":
        ax.text(
            0.02,
            0.98,
            text,
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
            fontsize=20,
            color=color,
        )
    elif pos == "out":
        ax.text(
            0.02,
            1.0,
            text,
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=ax.transAxes,
            fontsize=20,
            color=color,
        )
    return ax


def addEra(ax, lumi, era, energy="13 TeV") -> mpl.axis.Axis:
    text = f"${lumi}\\ \\mathrm{{fb}}^{{-1}}$ {era} ({energy})"

    ax.text(
        1,
        1,
        text,
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
        fontsize=20,
    )
    return ax


def addText(ax, x, y, text, **kwargs):
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=20,
        **kwargs,
    )
    return ax

def addCutTable(ax,cut_list,**kwargs):
    if 'loc' not in kwargs:
       kwargs["loc"] = 'upper center'
    num_of_sets = len(cut_list.keys())
    length = 0
    for key in cut_list:
        temp_length = len(cut_list[key])
        if temp_length > length:
            length = temp_length
    
    for key in cut_list:
        cut_list[key] = np.pad(cut_list[key],length-len(cut_list[key]),constant_values="")
    cuts_table = np.empty((length+1,num_of_sets),dtype=object)
    cuts_table[0] = np.array(list(cut_list.keys()))
    for index,key in enumerate(cut_list):
        cuts_table[1:,index] = cut_list[key]
    ax.table(cuts_table,cellLoc='center',**kwargs).scale(1,2)
     
    return ax
