#include <unordered_map>
#include <unordered_set>
#include <string>
#include <iostream>
#include <format>
#include <memory>
#include <fstream>
#include <filesystem>
#include "TFile.h"
#include "TTree.h"
#include "ROOT/RDataFrame.hxx"


std::filesystem::path getUniquePath(const std::filesystem::path& p){
    if(! std::filesystem::exists(p)) {return p;}
    auto ret = p;
    auto parent = p.parent_path();
    auto stem = p.stem();
    auto ext = p.extension();
    int i = 1;
    while(std::filesystem::exists(ret)){
        ret  = parent / (stem.string() + "_" + std::to_string(i)) ;
        ret.replace_extension(ext);
    }
    return ret;
}


std::vector<std::string> getFiles(){
    std::vector<std::string> file_paths;
    std::ifstream input("signal_files_2018.txt");
    std::string line;

    while(std::getline(input, line)){
        file_paths.push_back(std::string("root://cmsxrootd.fnal.gov//") + line);
    }
    return file_paths;

}

void processOneFile(const std::filesystem::path& infile, const std::filesystem::path& outdir){
    std::cout << std::format("Processing {} file to outdirectory {}\n", infile.string(), outdir.string());

    auto base = infile.stem().string();

    std::vector<std::string> gen_names;
    std::vector<std::string> other_names;

    ROOT::RDataFrame rdf("Events", infile.string());

    for(const auto name: rdf.GetColumnNames()){
        if(name.find("GenModel") == std::string::npos){
            other_names.push_back(name);
        } else {
            gen_names.push_back(name);

        }
    }

    for(const auto& name : gen_names){
        auto out_file_base = "signal" + name.substr(8,name.size()-1);
        auto try_out_path =  outdir / out_file_base;
        auto filtered = rdf.Filter([](bool x){return x;}, {name});
        const auto final_name = getUniquePath(outdir / base / ( std::string("signal") + name.substr(8,name.size()-1) + ".root"));
        std::filesystem::create_directories(final_name.parent_path());
        std::cout << std::format("Saving {} to {}\n", name, final_name.string());
        filtered.Snapshot("Events", final_name.string(), other_names);
    }
}


int main(int argc, char* argv[]) {
    if(argc != 3){
        return 1;
    }
    std::string fname(argv[1]);
    std::string outdir(argv[2]);
    processOneFile(fname, outdir);
    return 0;
}
