/* Setting up TPythia8 locally,
 * even if it was not configured
 * and built. This *does* require
 * that Pythia8 is installed
 * somewhere.
 */

#include "TROOT.h"
#include "TSystem.h"
#include "TInterpreter.h"
#include "TSystemDirectory.h"
#include "TObject.h"
#include "TObjArray.h"
#include "TList.h"
#include "TString.h"

// Check for the environment variables needed for TPythia8 functionality.
Bool_t PythiaCheck(){
    const Char_t *p8dataenv = gSystem->Getenv("PYTHIA8DATA");
    if (!p8dataenv) {
        const Char_t *p8env = gSystem->Getenv("PYTHIA8");
        if (!p8env) {
//            Error("generator.C",
//                  "Environment variable PYTHIA8 must contain path to pythia directory!");
            return kFALSE;
        }
        TString p8d = p8env;
        p8d += "/xmldoc";
        gSystem->Setenv("PYTHIA8DATA", p8d);
    }
    const Char_t* path = gSystem->ExpandPathName("$PYTHIA8DATA");
    if (gSystem->AccessPathName(path)) {
//        Error("generator.C",
//              "Environment variable PYTHIA8DATA must contain path to $PYTHIA8/xmldoc directory!");
        return kFALSE;
    }
    return kTRUE;
}
 // Check for the ROOT TPythia8 library.
Bool_t TPythiaCheck(){
    // avoid the annoying error messages
    gROOT->ProcessLine("gErrorIgnoreLevel = kFatal;");
    // Try to load the TPythia8 library.
    Int_t status = gSystem->Load("libEGPythia8");
    // return error reporting to the default state
    gROOT->ProcessLine("gErrorIgnoreLevel = kError;");
    if(status == 0) return kTRUE;
    else return kFALSE;
}

// TODO: Simplify things -- there are only 2 likely cases: TPythia8 set up fully or not at all.
void rootlogon() {
    
    std::cout << "Running local rootlogon.C." << std::endl;
    std::cout << "Checking if TPythia8 is configured with your ROOT build." << std::endl;
    Bool_t pythia_setup = PythiaCheck();
    Bool_t tpythia_setup = TPythiaCheck();
    
    // Print statements for Pythia env variables check
    if(pythia_setup == kFALSE) std::cout << "\nEnvironment variables $PYTHIA8 and/or $PYTHIA8DATA are not set up." << std::endl;
    else{
        std::cout << "\nFound $PYTHIA8 and $PYTHIA8DATA." << std::endl;
        std::cout << "\t$PYTHIA8 \t= " << gSystem->Getenv("PYTHIA8") << std::endl;
        std::cout << "\t$PYTHIA8DATA \t= " << gSystem->Getenv("PYTHIA8DATA") << std::endl;
    }

    // Print statements for TPythia8 library check
    if(tpythia_setup == kFALSE) std::cout << "\nDid not find libEGPythia8, TPythia8 is not set up -> Will use local TPythia8." << std::endl;
    else std::cout << "\nFound libEGPythia8, TPythia8 is set up." << std::endl;
    
    // If everything is set up, we can finish here.
    if(pythia_setup && tpythia_setup) return;
    
    
    // Now we explicitly take care of the env variables.
    // We assume that there is *some* path to the
    // pythia install directory in $LD_LIBRARY_PATH.
    // This is a safe assumption on the UChicago Tier3
    // (and presumably other ATLAS computing env's) if
    // both ROOT & Pythia8 have been set up using
    // lsetup/lcgenv.
    TString PYTHIA8 = gSystem->Getenv("PYTHIA8");
    TString PYTHIA8DATA = gSystem->Getenv("PYTHIA8DATA");
    TString PYTHIA8_LIB_DIR = PYTHIA8 + "/lib";

    // 1) If PYTHIA8 and/or PYTHIA8DATA are not found
    if(!pythia_setup){
        TString pythia_substring = "pythia8"; // substring we will search for in LD_LIBRARY_PATH & DYLD_LIBRARY_PATH (dyld for macOS)
        TString LD_LIBRARY_PATH = TString(gSystem->Getenv("LD_LIBRARY_PATH"));
        TString DYLD_LIBRARY_PATH = TString(gSystem->Getenv("DYLD_LIBRARY_PATH"));
        TString LIBRARY_PATH = LD_LIBRARY_PATH + ":" + DYLD_LIBRARY_PATH;
        TObjArray* path_arr = LIBRARY_PATH.Tokenize(":");
        
        for (UInt_t i = 0; i < path_arr->GetEntries(); i++) {
            TString entry = ((TObjString*)path_arr->At(i))->String();
            if(entry.Contains(pythia_substring, TString::kIgnoreCase)){
                PYTHIA8_LIB_DIR = TString(entry);
                break;
            }
        }
        if(PYTHIA8_LIB_DIR.EqualTo("")){
            Error("rootlogon.C", "Did not find any reference to pythia8 in $(DY)LD_LIBRARY_PATH. Aborting.");
            return;
        }
        if(!(TString(PYTHIA8_LIB_DIR(PYTHIA8_LIB_DIR.Length()-4,PYTHIA8_LIB_DIR.Length()-1)).EqualTo("/lib"))){
            Error("rootlogon.C", "Pythia8 path in $(DY)LD_LIBRARY_PATH does not end with \"/lib\". Aborting.");
            return;
        }
        // To get PYTHIA8DATA from PYTHIA8_LIB_DIR, we should be able
        // to strip off the "/lib" that (should) occur at the end.
        // TODO: Should we use TString::ReplaceAll() or strip a fixed-length substring off the end?
        PYTHIA8DATA = TString(PYTHIA8_LIB_DIR).ReplaceAll("/lib","");
        PYTHIA8 = PYTHIA8DATA;
        gSystem->Setenv("PYTHIA8",PYTHIA8);
        std::cout << "\nSet $PYTHIA8 = \t" << PYTHIA8 << std::endl;

        PYTHIA8DATA.Append("/share/Pythia8/xmldoc");
        gSystem->Setenv("PYTHIA8DATA",PYTHIA8DATA);
        std::cout << "Set $PYTHIA8DATA = \t" << PYTHIA8DATA << std::endl;
        delete path_arr;
    }
    
    // If TPythia8 isn't set up at all, we only really need $PYTHIA8.
    // From there, we can get the include files, as well as the shared library.
    TString PYTHIA8_INCLUDE = TString(PYTHIA8) + "/include";
    if(!tpythia_setup){
        gSystem->AddIncludePath("-I" + PYTHIA8_INCLUDE);
        std::cout << "\nAdded to include path via gSystem:\t" << PYTHIA8_INCLUDE << std::endl;
        
        // Lastly, we need to load the pythia8 shared library.
        // we will check for any of them and use the first one we find.
        TSystemDirectory* sysDir = new TSystemDirectory("pyth_lib",PYTHIA8_LIB_DIR);
        TList* lib_files = sysDir->GetListOfFiles();
        TString PYTHIA8_LIB = "";
        
        const UInt_t n_filetypes = 3;
        TString filetypes [n_filetypes] = {".so",".a",".dylib"};
        for (UInt_t i = 0; i < lib_files->GetEntries(); i++) {
            TString entry = ((TObjString*)lib_files->At(i))->String();
            for (UInt_t j = 0; j < n_filetypes; j++) {
                if(entry.Contains(filetypes[j],TString::kIgnoreCase)){
                    PYTHIA8_LIB = TString(entry);
                    break;
                }
            }
            if(!PYTHIA8_LIB.EqualTo("")) break;
        }
        
        if(PYTHIA8_LIB.EqualTo("")){
            Error("rootlogon.C", "Did not find the Pythia8 shared library. Aborting.");
            return;
        }
        
        // Now we will add PYTHIA8_LIB_DIR to the dynamic path. Alternatively one can
        // just use the full path, /path/to/lib/library.[so|lib|dylib|a],
        // in TSystem::Load(). (The full path is not needed if you already have
        // the Pythia8 library in $(DY)LD_LIBRARY_PATH, but this might still cause
        // issues when using PyROOT.)
        
        TString dynamic_path = gSystem->GetDynamicPath();
        dynamic_path = PYTHIA8_LIB_DIR + ":" + dynamic_path;
        gSystem->SetDynamicPath(dynamic_path);
        std::cout << "\nAdded to dynamic path via gSystem:\t" << PYTHIA8_LIB_DIR << std::endl;
        std::cout << "\nFound Pythia8 shared library:\t" << PYTHIA8_LIB_DIR << "/" << PYTHIA8_LIB << std::endl;
        gSystem->Load(PYTHIA8_LIB);
        std::cout << "Loaded " << PYTHIA8_LIB << "." << std::endl;
        
        // Finally, we are ready to load our local TPythia8.
        // The result is that for any following code in
        // C++/ROOT or PyROOT, TPythia8 will function normally.
        Int_t status = gSystem->CompileMacro("TPythia8.cxx");
        std::cout << "Compiled & loaded local TPythia8.cxx." << std::endl;
    }
    return;
}
