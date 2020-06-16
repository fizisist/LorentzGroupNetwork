//  File: reduction.C
//  Author: Jan Offermann
//  Date: 07/01/19.
//  Goal: A bunch of different plotting tools for reviewing data.
//

#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <stdio.h>

#include "TSystem.h"
#include "TSystemDirectory.h"
#include "TStyle.h"
#include "TString.h"
#include "TFile.h"
#include "TTree.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TTreeReaderArray.h"
#include "TChain.h"
#include "TParticle.h"
#include "TClonesArray.h"
#include "TList.h"
#include "TObject.h"
#include "TObjString.h"
#include "TParameter.h"
#include "TLorentzVector.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TH1D.h"
#include "TH1I.h"
#include "TH2D.h"
#include "THStack.h"
#include "TGraph.h"
#include "TLegend.h"
#include "TLorentzVector.h"
#include "TPaveText.h"
#include "TMarker.h"
#include "TEllipse.h"
#include "TAxis.h"

// helper function -- makes a TLegend with some good default params
TLegend* SetupLegend(Double_t x1, Double_t y1, Double_t x2, Double_t y2){
    TLegend* leg = new TLegend(x1,y1,x2,y2);
    leg->SetFillColor(0);
    leg->SetFillStyle(0); // make it transparent!
    leg->SetBorderSize(0);
    leg->SetTextFont(42);
    leg->SetTextAngle(0);
    leg->SetTextColor(kBlack);
    leg->SetTextSize(0.05);
    leg->SetTextAlign(12);
    return leg;
}

// helper function -- makes a TPaveText with some good default params
TPaveText* SetupPave(Double_t x1, Double_t y1, Double_t x2, Double_t y2){
    TPaveText* pave = new TPaveText(x1,y1,x2,y2, "NDC");
    pave->SetTextFont(42);
    pave->SetTextAngle(0);
    pave->SetTextColor(kBlack);
    pave->SetTextSize(0.04);
    pave->SetTextAlign(12);
    pave->SetFillStyle(0);
    pave->SetBorderSize(0);
    return pave;
}

// chain together TTree's named "tree_name" from all ROOT files in a given directory
TChain* GetChain(TString directory, TString tree_name){
    TChain* chain = new TChain(tree_name, tree_name);
    std::vector<TString> root_files;
    TSystemDirectory* dir = new TSystemDirectory("file_dir", directory);
    TList* list_of_files = dir->GetListOfFiles(); // get a TList w/ list of directory contents, as (abstract) TObject's
    for (ULong_t i = 0; i < list_of_files->GetEntries(); i++) {
        TString filename = ((TObjString*)list_of_files->At(i))->String().Prepend("/").Prepend(directory); // adding filepath
        if(filename.Contains(".root")) root_files.append(filename);
    }
    for (ULong_t i = 0; i < root_files.size(); i++) chain->Add(root_files.at(i));
    delete dir;
    return chain;
}

// make a Lego plot of the jet constituents (COLZ option for TH2 would also be good, heat maps are good in general!)
void JetPlotBinned(Int_t nconst, Double_t* eta, Double_t* phi, Double_t* pt, Int_t sig = 0, Float_t offset = 1.){
    
    /*
     * Inputs:
     * nconst = number of jet constituents (length of eta, phi, pt)
     * eta = array of eta values for jet constituents (x-axis)
     * phi = array of phi values for jet constituents (y-axis)
     * pt = array of pt values for jet constituents (z-axis)
     * sig = signal (1) / background (0) flag
     * offset = title offset (for formatting purposes)
     */
    
    // some formatting for the plot -- these things are hardcoded at the moment
    Double_t x_min = -3.15;
    Double_t x_max = -1. * x_min;
    Double_t y_min = 0.;
    Double_t y_max = x_max - x_min;
    Int_t nbins = (Int_t)(y_max / 0.1);
    
    // some hardcoded labels for signal & background processes, if needed
    TString signal_label = "t#bar{t}"; // ttbar for example (demonstrating TLatex usage)
    TString bckgd_label = "QCD"; // QCD, for example

    TH2D* constituents = new TH2D("jet_const", "Jet Constituents;#eta;#phi;p_{T} (GeV)", nbins, x_min, x_max, nbins, y_min, y_max);
    constituents->GetXaxis()->SetTitleOffset(offset);
    constituents->GetYaxis()->SetTitleOffset(offset);
    constituents->GetZaxis()->SetTitleOffset(offset);
    for (Int_t i = 0; i < nconst; i++) constituents->Fill(eta[i],phi[i],pt[i]);
    TCanvas* canv = new TCanvas("c1", "c1", 800, 600);
    canv->cd();
    constituents->Draw("lego2");
    canv->Update();

    // add a textbox saying whether or not this is signal
    TPaveText* pave = SetupPave(0.75,0.875,0.95,0.975);
    TString process = "";
    if(sig == 0) process.Append(signal_label);
    else process.Append(bckgd_label);
    pave->AddText(TString("process = ").Append(process));

    pave->Draw();
    gStyle->SetOptStat(0); // remove default histogram stat box -- useful for debugging but kind of annoying otherwise
    canv->Draw();
    return;
}


// unbinned jet plot -- probably most useful for debugging since actual detectors are always binned
void JetPlotUnbinned(Int_t nconst, Double_t* eta, Double_t* phi, Double_t jeta, Double_t jphi, Double_t jet_radius, Double_t eta_max, Int_t sig = 0, Double_t tphi = 0., Double_t teta = 0.){
    
    /*
     * Inputs:
     * nconst = number of jet constituents (length of arrays)
     * eta = eta values of constituents
     * phi = phi values of constituents
     * jeta = eta value of the full jet (NOT constituents)
     * jphi = eta value of the full jet
     * jet_radius = radius of jet
     * eta_max = maximum eta for jet-finder (adds lines to plot to show cutoffs)
     * sig = signal(1) / background(0) flag
     * tphi = phi value of truth-level jet mother particle (e.g. truth-level top for top quark jet)
     * teta = eta value of truth_level jet mother particle
     */
    
    // make a scatter plot of the jet constituents, do some formatting
    TGraph* constituents = new TGraph(nconst, eta, phi);
    constituents->GetXaxis()->SetLimits(-TMath::Pi(), TMath::Pi());
    constituents->GetHistogram()->SetMinimum(0.);
    constituents->GetHistogram()->SetMaximum(2. * TMath::Pi());
    constituents->SetTitle("jet constituents");
    constituents->SetMarkerColor(kBlue);
    constituents->SetLineColor(0);
    constituents->SetMarkerStyle(kStar);
    TCanvas* canv = new TCanvas("c1", "c1", 800, 800);
    canv->cd();
    constituents->Draw("ap");
    constituents->GetXaxis()->SetTitle("#eta");
    constituents->GetYaxis()->SetTitle("#phi");
    canv->Update();
    
    // plot the center of the jet
    jphi = jphi - 2. * TMath::Pi() * TMath::Floor(jphi / (2. * TMath::Pi())); // mod 2 pi
    TMarker* jet = new TMarker(jeta, jphi, kFullCircle);
    jet->SetMarkerColor(kRed);
    jet->Draw("same");
    TEllipse* jet_circle = new TEllipse(jeta, jphi, jet_radius, jet_radius);
    jet_circle->SetLineColor(kRed);
    jet_circle->SetFillStyle(0);
    jet_circle->Draw("same");
    
    // add a textbox saying whether or not this is signal
    TPaveText* pave = SetupPave(0.75,0.85,0.95,0.95);
    pave->AddText(TString("signal = ").Append(std::to_string(sig)));
    
    // if this is a signal event, also overlay the eta/phi of the truth-level top
    if(sig == 1){
        TMarker* sig = new TMarker(teta, tphi, kFullStar);
        sig->SetMarkerColor(kGreen);
        sig->Draw("same");
    }
    TLine* l1 = new TLine(-eta_max, 0., -eta_max, 2. * TMath::Pi());
    TLine* l2 = new TLine(eta_max, 0., eta_max, 2. * TMath::Pi());
    l1->Draw();
    l2->Draw();
    pave->Draw();
    canv->Draw();
    return;
}

// plot jet at index "event_index" of the TChain of TTree's called "tree_name" in all ROOT files in directory "dir"
void JetPlot(TString dir, TString tree_name, Long64_t event_index, Int_t mode = 0, Float_t offset = 1.5){
    //mode = 0: unbinned plot
    //mode = 1: binned plot
    
    Double_t jet_radius = 0.8;
    Double_t eta_max = 2.0;
    TChain* chain = GetChain(dir, tree_name);
    Long64_t nentries = chain->GetEntries();
    if(event_index >= nentries){
        std::cout << "Warning: event_index > nentries = " << nentries << std::endl;
        return;
    }
    
    // --- TTree Reading: this section will depend on how your data is stored ---
    // variables for getting things from the TTree/TChain
    Double_t E[200];
    Double_t px[200];
    Double_t py[200];
    Double_t pz[200];
    Int_t sig = 0.;
    Double_t tE = 0.;
    Double_t tpx = 0.;
    Double_t tpy = 0.;
    Double_t tpz = 0.;
    Double_t teta = 0.;
    Double_t tphi = 0.;
    Double_t jeta = 0.;
    Double_t jphi = 0.;
    // setting branch addresses for the variables above - could also use TTreeReader + TTreeReaderValue & TTreeReaderArray in principle
    chain->SetBranchAddress("is_signal", &sig);
    chain->SetBranchAddress("E", &E);
    chain->SetBranchAddress("px", &px);
    chain->SetBranchAddress("py", &py);
    chain->SetBranchAddress("pz", &pz);
    if(mode == 0){
        chain->SetBranchAddress("truth_E", &tE);
        chain->SetBranchAddress("truth_px", &tpx);
        chain->SetBranchAddress("truth_py", &tpy);
        chain->SetBranchAddress("truth_pz", &tpz);
        chain->SetBranchAddress("jet_eta", &jeta);
        chain->SetBranchAddress("jet_phi", &jphi);
    }
    
    // --- End of TTree Reading setup ---

    // select the entry # as given by the user
    chain->GetEntry(event_index); // for TTreeReader usage, replace with TTreeReader::SetEntry()
    // explicitly count the # of non-zero jet constituents -- the TTree being read has zero-padded array branches
    Int_t nconst_temp = 0;
    for (Int_t i = 0; i < 200; i++) {
        if(E[i] <= 0.) break;
        nconst_temp++;
    }
    
    // get the eta and phi for each constituent
    const Int_t nconst = nconst_temp;
    Double_t eta[nconst];
    Double_t phi[nconst];
    Double_t pt[nconst];
    for (Int_t i = 0; i < nconst; i++) {
        TLorentzVector* vec = new TLorentzVector(); // will use built-in TLorentzVector conversions
        vec->SetPxPyPzE(px[i],py[i],pz[i],E[i]);
        eta[i] = vec->Eta();
        phi[i] = vec->Phi();
        phi[i] = phi[i] - 2. * TMath::Pi() * TMath::Floor(phi[i] / (2. * TMath::Pi())); // mod 2 pi
        pt[i] = vec->Pt();
        delete vec;
    }
    
    // if this is a signal event, also overlay the eta/phi of the truth-level top
    if(sig == 1){
        TLorentzVector* vec = new TLorentzVector();
        vec->SetPxPyPzE(tpx,tpy,tpz,tE);
        teta = vec->Eta();
        tphi = vec->Phi(); // mod 2 pi
        tphi = tphi - 2. * TMath::Pi() * TMath::Floor(tphi / (2. * TMath::Pi())); // mod 2 pi
    }
    
    if(mode == 0){
        if(sig == 0) JetPlotUnbinned(nconst, eta, phi, jeta, jphi, jet_radius, eta_max);
        else JetPlotUnbinned(nconst, eta, phi, jeta, jphi, jet_radius, eta_max, sig, teta, tphi);
    }
    else JetPlotBinned(nconst, eta, phi, pt, sig, offset);
    return;
}

