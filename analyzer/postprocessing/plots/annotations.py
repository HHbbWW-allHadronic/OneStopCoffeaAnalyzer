import mplhep


def addCMSBits(ax, sectors):
    lumis = set(str(x.sector_params.dataset.lumi) for x in sectors)
    energies = set(str(x.sector_params.dataset.era.energy) for x in sectors)
    lumi_text = f"{'/'.join(lumis)} fb$^{{-1}}$ ({'/'.join(energies)} TeV)"
    mplhep.cms.lumitext(text=lumi_text, ax=ax)
    mplhep.cms.text(text="Preliminary", ax=ax)
