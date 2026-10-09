from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

events = NanoEventsFactory.from_root(
    {"root://cmseos.fnal.gov//eos/uscms/store/user/gavetter/MYOMC/ggtoHHto2B2W_HADRONIC_MERGED/output1.root": "Events"},
    schemaclass=NanoAODSchema,
    entry_stop=10_000,
    mode = "eager",
).events()

print(events.fields)            # top-level collections
print(events.FatJet.globalParT3_XWW3q)       # attributes on the Muon collection
