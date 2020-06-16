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

void PlotFromRawFile(TString path_to_file = "../samples_root/raw/particle/noMPI/Top_400-600_13TeV.root", Bool_t isHadronized = kTRUE, Bool_t isSignal = kTRUE){
    TFile* f = new TFile(path_to_file, "READ");
    TTree* t = (TTree*)f->Get("events");
    const Long64_t nentries = t->GetEntries();
    
    // Get the # of jet constituents to be saved, from a TParameter in the events TTree
    TList* userinfo = t->GetUserInfo();
    UInt_t nentries_ui = userinfo->GetEntries();
    Int_t nconst_temp = 0;
    Double_t jet_radius = 0.;
    for (UInt_t i = 0; i < nentries_ui; i++) {
        TParameter<Int_t>* par = ((TParameter<Int_t>*)(userinfo->At(i)));
        TParameter<Double_t>* par_d = ((TParameter<Double_t>*)(userinfo->At(i)));
        if (TString(par->GetName()).EqualTo("njet_constituents")) nconst_temp = par->GetVal();
        else if (TString(par_d->GetName()).EqualTo("jet_radius")) jet_radius = par_d->GetVal();
    }
    const Int_t nconst = nconst_temp;
    
    // set up storage
    TString path_to_output = TString(path_to_file).ReplaceAll("/raw/", "/reduced/");
    if(path_to_output.EqualTo(path_to_file)){
        std::cout << "Error: output path matches input path. Ending." << std::endl;
        f->Close();
        return;
    }
    Int_t is_signal;
    TParticle* part;
    TParticle* part2;

    //set up reading
    TTreeReader myReader(t);
    TTreeReaderValue<Int_t> rv_proc_id(myReader, "process_id");
    TTreeReaderValue<Int_t> rv_njet(myReader, "njet");
    TTreeReaderValue<Int_t> rv_njet_const(myReader, "njet_const"); // length of jet_const branch. This is a flattened 2D array of jets' constituents' indices
    TTreeReaderArray<Double_t> ra_jet_m(myReader, "jet_m");
    TTreeReaderArray<Double_t> ra_jet_pt(myReader, "jet_pt");
    TTreeReaderArray<Double_t> ra_jet_eta(myReader, "jet_eta");
    TTreeReaderArray<Double_t> ra_jet_phi(myReader, "jet_phi");
    TTreeReaderArray<Int_t> ra_jet_const(myReader, "jet_const");
    TTreeReaderValue<TParticle> rv_truth(myReader, "truth"); // top
    TTreeReaderValue<TParticle> rv_truth2(myReader, "truth2"); // anti-top
    TTreeReaderValue<Int_t> rv_is_truth(myReader, "is_truth");
    //we will not use TTreeReaderArray with TClonesArray
    TClonesArray *arr = new TClonesArray("TParticle");
    t->GetBranch("event")->SetAutoDelete(kFALSE);
    t->SetBranchAddress("event", &arr);
    // for the top
    Double_t dR_array [nentries];
    Double_t eta_array [nentries];
    Double_t pt_array [nentries];
    Double_t phi_array [nentries];
    // for the anti-top
    Double_t dR_array2 [nentries];
    Double_t eta_array2 [nentries];
    Double_t pt_array2 [nentries];
    Double_t phi_array2 [nentries];
    
//    std::cout << "jet_radius = " << jet_radius << std::endl;
    for (Long64_t i = 0; i < nentries; i++) {
        
        dR_array[i] = -999.;
        eta_array[i] = -999.;
        pt_array[i] = -999.;
        phi_array[i] = -999.;
        dR_array2[i] = -999.;
        eta_array2[i] = -999.;
        pt_array2[i] = -999.;
        phi_array2[i] = -999.;
        
        myReader.SetEntry(i); // set ready entry
        arr->Clear();
        t->GetEntry(i); // explicitly set tree entry for arr
        Int_t n = arr->GetEntriesFast(); // number of particles
        Int_t njet = *rv_njet;
        Int_t njet_const = *rv_njet_const;
        Int_t closest_jet_index = -1;
        Double_t deltaR2 = -1.;
        /*
         * Strategy:
         * For hadronized events, we do jet-matching: For signal (Top), we pick the jet nearest
         * the parton-level top in eta/phi. For background, we simply pick the highest-pt jet.
         *
         * For non-hadronized events, we simply record the parton level Top or highest-pt particle.
         * This corresponds with a single particle, so the list of 200 highest-pt constituents will
         * be empty save for one entry.
         */
        if(isHadronized){
            if(isSignal){ // hadronized signal
                // Find the constituents of the jet that matches closest with the parton-level top
                // 1) Get the top
                if(*rv_is_truth != 1){
                    std::cout << "Uh oh, signal event w/out truth top. Skipping!" << std::endl;
                    continue;
                }
                // top
                TParticle top = *rv_truth;
                Double_t top_m = top.GetCalcMass();
                Double_t top_eta = top.Eta();
                Double_t top_phi = top.Phi();
                Double_t top_pt = top.Pt();
                TLorentzVector top_vec = TLorentzVector();
                top_vec.SetPtEtaPhiM(top_pt, top_eta, top_phi, top_m);
                // anti-top
                TParticle atop = *rv_truth2;
                Double_t atop_m = atop.GetCalcMass();
                Double_t atop_eta = atop.Eta();
                Double_t atop_phi = atop.Phi();
                Double_t atop_pt = atop.Pt();
                TLorentzVector atop_vec = TLorentzVector();
                top_vec.SetPtEtaPhiM(atop_pt, atop_eta, atop_phi, atop_m);
                // 2) Find the index of the jet closest to the top
                for (Int_t j = 0; j < njet; j++) {
                    
                    Double_t jet_m = ra_jet_m[j];
                    Double_t jet_pt = ra_jet_pt[j];
                    Double_t jet_eta = ra_jet_eta[j];
                    Double_t jet_phi = ra_jet_phi[j];
                    TLorentzVector jet = TLorentzVector();
                    jet.SetPtEtaPhiM(jet_pt, jet_eta, jet_phi, jet_m);
                    Double_t deltaR2_temp = jet.DeltaR(top_vec) * jet.DeltaR(top_vec);
                    if(deltaR2_temp < 0.) std::cout << "Agh, found deltaR^2 < 0." << std::endl;
                    if((deltaR2_temp < deltaR2 || deltaR2 < 0.) /*&& deltaR2_temp < jet_radius * jet_radius */){
                        deltaR2 = deltaR2_temp;
                        closest_jet_index = j;
                    }
                }
                if(closest_jet_index == -1){
                    std::cout << "Uh oh, closest jet not found." << std::endl;
                    continue;
                }
                dR_array[i] = TMath::Sqrt(deltaR2);
                eta_array[i] = top_eta;
                pt_array[i] = top_pt;
                phi_array[i] = top_phi;
                
                // 3) Find the distance between the jet closest to the top, and the antitop
                Double_t jet_m = ra_jet_m[closest_jet_index];
                Double_t jet_pt = ra_jet_pt[closest_jet_index];
                Double_t jet_eta = ra_jet_eta[closest_jet_index];
                Double_t jet_phi = ra_jet_phi[closest_jet_index];
                TLorentzVector jet = TLorentzVector();
                jet.SetPtEtaPhiM(jet_pt, jet_eta, jet_phi, jet_m);
                deltaR2 = jet.DeltaR(atop_vec) * jet.DeltaR(atop_vec);
                dR_array2[i] = TMath::Sqrt(deltaR2);
                eta_array2[i] = atop_eta;
                pt_array2[i] = atop_pt;
                phi_array2[i] = atop_phi;
            }
        }
    }
    f->Close();
    
    TH1D* hist = new TH1D("hist", "#Delta R(x, j^{t}_{match});#Delta R;Counts", 200,0.,5.);
    TH1D* hist2 = new TH1D("hist2", "#Delta R(#bar{t}, j^{matched});#Delta R;Counts", 200,0.,5.);
    TH2D* hist_eta = new TH2D("hist_eta", "Min. #Delta R(t, j) vs. #eta(t);#eta(t);#Delta R", 100,-5.,5.,200,0.,5.);
    TH2D* hist_pt = new TH2D("hist_pt", "Min. #Delta R(t, j) vs. p_{T}(t);pt_{t};#Delta R", 2000,0.,4000.,200,0.,5.);
    TH2D* hist_tjtbar = new TH2D("hist_tjtbar", "#Delta R(#bar{t}, j^{t}_{match}) vs. Delta R(t, j^{t}_{match});#Delta R(t, j^{t}_{match});#Delta R(#bar{t}, j^{t}_{match})", 100,0.,5.,100,0.,5.);
    TH2D* hist_eta1eta2 = new TH2D("hist_eta1eta2", "#eta(#bar{t}) vs. #eta(t);#eta(t);#eta(#bar{t})", 100,-5.,5.,100,-5.,5.);
    TH2D* hist_phi1phi2 = new TH2D("hist_phi1phi2", "#phi(#bar{t}) vs. #phi(t);#phi(t);#phi(#bar{t})", 200,0.,6.5,200,0.,6.5);
    hist->SetLineColor(kBlue);
    hist2->SetLineColor(kRed);
    TLegend* leg = SetupLegend(0.75,0.7,0.85,0.8);
    leg->AddEntry(hist, "x=t", "lf");
    leg->AddEntry(hist2, "x=#bar{t}", "lf");
    
    for (Long64_t i = 0; i < nentries; i++){
        hist->Fill(dR_array[i]);
        hist2->Fill(dR_array2[i]);
        hist_eta->Fill(eta_array[i],dR_array[i]);
        hist_pt->Fill(pt_array[i],dR_array[i]);
        hist_tjtbar->Fill(dR_array[i],dR_array2[i]);
        hist_eta1eta2->Fill(eta_array[i], eta_array2[i]);
        hist_phi1phi2->Fill(phi_array[i], phi_array2[i]);
    }
    TCanvas* c1 = new TCanvas("c1", "c1", 800, 600);
    TCanvas* c2 = new TCanvas("c2", "c2", 800, 600);
    TCanvas* c3 = new TCanvas("c3", "c3", 800, 600);
    TCanvas* c4 = new TCanvas("c4", "c4", 800, 600);
    TCanvas* c5 = new TCanvas("c5", "c5", 800, 600);
    c5->cd();
    TPad* pad1 = new TPad("p1", "p1", 0.,0.,0.5,1.);
    TPad* pad2 = new TPad("p2", "p2", 0.5,0.,1.,1.);
    gStyle->SetOptStat(0);
    
    c1->cd();
    hist->Draw("HIST");
    hist2->Draw("HIST SAME");
    leg->Draw();
    c2->cd();
    hist_eta->Draw("COLZ");
    c3->cd();
    hist_pt->Draw("COLZ");
    c4->cd();
    hist_tjtbar->Draw("COLZ");
    c5->cd();
    pad1->cd();
    hist_eta1eta2->Draw("COLZ");
    pad2->cd();
    hist_phi1phi2->Draw("COLZ");
    c1->Draw();
    c2->Draw();
    c3->Draw();
    c4->Draw();
    c5->cd();
    pad1->Draw();
    pad2->Draw();
    c5->Draw();
    TString yada = ((TObjString*)(TString(path_to_file).Tokenize("/")->Last()))->String();
    yada.ReplaceAll(".root", "XXX.eps");
    TString name1 = TString(yada).ReplaceAll("XXX", "-1");
    TString name2 = TString(yada).ReplaceAll("XXX", "-2");
    TString name3 = TString(yada).ReplaceAll("XXX", "-3");
    TString name4 = TString(yada).ReplaceAll("XXX", "-4");
    TString name5 = TString(yada).ReplaceAll("XXX", "-5");
    c1->SaveAs(name1);
    c2->SaveAs(name2);
    c3->SaveAs(name3);
    c4->SaveAs(name4);
    c5->SaveAs(name5);
    return;
}

TChain* GetChain(TString directory){
    
    TString files [3] = {TString(directory).Append("/out_test.root"), TString(directory).Append("/out_train.root"), TString(directory).Append("/out_valid.root")};
    TChain* chain = new TChain("events_reduced", "events_reduced");
    for (Int_t i = 0; i < 3; i++) chain->Add(files[i]);
    return chain;
}

// Get a histogram of the energy of jet constituents
TH1D* ConstituentEnergyHist(TString title, TString directory, TString hist_name, Int_t is_signal = 1){
    Int_t sig;
    Int_t n;
    Double_t E[200];
    TChain* chain = GetChain(directory);
    Long64_t nentries = chain->GetEntries();
    chain->SetBranchAddress("is_signal", &sig);
    chain->SetBranchAddress("n", &n); // this will always be the max length, 200
    chain->SetBranchAddress("E", &E);
    
    TH1D* hist = new TH1D(hist_name, title, 250, 0, 5000); // empirically determined maximum
    
    for (Long64_t i = 0; i < nentries; i++) {
        chain->GetEntry(i);
        if(sig==is_signal){
            for (Int_t j = 0; j < n; j++) {
                if(E[j] <= 0.) break;
                hist->Fill(E[j]);
            }
        }
    }
    delete chain;
    return hist;
}

// Get a histogram of the average energy of jet constituents per event
TH1D* AvgConstituentEnergyHist(TString title, TString directory, TString hist_name, Int_t is_signal = 1){
    Int_t sig;
    Int_t n;
    Double_t E[200];
    TChain* chain = GetChain(directory);
    Long64_t nentries = chain->GetEntries();
    chain->SetBranchAddress("is_signal", &sig);
    chain->SetBranchAddress("n", &n); // this will always be the max length, 200
    chain->SetBranchAddress("E", &E);
    
    TH1D* hist = new TH1D(hist_name, title, 250, 0, 5000); // empirically determined maximum
    
    for (Long64_t i = 0; i < nentries; i++) {
        chain->GetEntry(i);
        if(sig==is_signal){
            Double_t avg_E = 0.;
            Double_t actual_n = 0.;
            for (Int_t j = 0; j < n; j++) {
                if(E[j] <= 0.) break;
                avg_E += E[j];
                actual_n += 1.;
            }
            avg_E = avg_E / actual_n;
            hist->Fill(avg_E);
        }
    }
    delete chain;
    return hist;
}

// Get a histogram of the difference between maximum and minimum energy of jet constituents
TH1D* DiffConstituentEnergyHist(TString title, TString directory, TString hist_name, Int_t is_signal = 1){
    Int_t sig;
    Int_t n;
    Double_t E[200];
    Double_t min_E;
    Double_t max_E;
    TChain* chain = GetChain(directory);
    Long64_t nentries = chain->GetEntries();
    chain->SetBranchAddress("is_signal", &sig);
    chain->SetBranchAddress("n", &n); // this will always be the max length, 200
    chain->SetBranchAddress("E", &E);
    
    TH1D* hist = new TH1D(hist_name, title, 250, 0, 5000); // empirically determined maximum
    
    for (Long64_t i = 0; i < nentries; i++) {
        chain->GetEntry(i);
        if(sig==is_signal){
            min_E = -1.;
            max_E = -1.;
            for (Int_t j = 0; j < n; j++) {
                if(E[j] <= 0.) break;
                if(E[j] < min_E || min_E < 0.) min_E = E[j];
                else if (E[j] > max_E) max_E = E[j];
            }
            hist->Fill(max_E - min_E);
        }
    }
    delete chain;
    return hist;
}

// Get a histogram of the mass of jets
TH1D* MassHist(TString title, TString directory, TString hist_name, Int_t is_signal = 1){
    Int_t sig;
    Double_t jet_m;
    TChain* chain = GetChain(directory);
    Long64_t nentries = chain->GetEntries();
    chain->SetBranchAddress("is_signal", &sig);
    chain->SetBranchAddress("jet_m", &jet_m);
    
    TH1D* hist = new TH1D(hist_name, title, 250, 0, 250.); // empirically determined maximum
    
    for (Long64_t i = 0; i < nentries; i++) {
        chain->GetEntry(i);
        if(sig==is_signal) hist->Fill(jet_m);
    }
    delete chain;
    return hist;
}


// Get a histogram of the mass of jet constituents
TH1D* ConstituentMassHist(TString title, TString directory, TString hist_name, Int_t is_signal = 1){
    Int_t sig;
    Int_t n;
    Double_t E[200];
    Double_t px[200];
    Double_t py[200];
    Double_t pz[200];
    TChain* chain = GetChain(directory);
    Long64_t nentries = chain->GetEntries();
    chain->SetBranchAddress("is_signal", &sig);
    chain->SetBranchAddress("n", &n); // this will always be the max length, 200
    chain->SetBranchAddress("E", &E);
    chain->SetBranchAddress("px", &px);
    chain->SetBranchAddress("py", &py);
    chain->SetBranchAddress("pz", &pz);
    
    TH1D* hist = new TH1D(hist_name, title, 60, 0, 6.); // empirically determined maximum
    
    TLorentzVector* vec = new TLorentzVector();
    for (Long64_t i = 0; i < nentries; i++) {
        chain->GetEntry(i);
        if(sig==is_signal){
            for (Int_t j = 0; j < n; j++) {
                if(E[j] <= 0.) break;
//                vec = new TLorentzVector();
                vec->SetPxPyPzE(px[j], py[j], pz[j], E[j]);
                hist->Fill(vec->M());
            }
        }
    }
    delete chain;
    delete vec;
    return hist;
}

// Get a histogram of the pT of jet constituents
TH1D* ConstituentPTHist(TString title, TString directory, TString hist_name, Int_t is_signal = 1){
    Int_t sig;
    Int_t n;
    Double_t E[200];
    Double_t px[200];
    Double_t py[200];
    TChain* chain = GetChain(directory);
    Long64_t nentries = chain->GetEntries();
    chain->SetBranchAddress("is_signal", &sig);
    chain->SetBranchAddress("n", &n); // this will always be the max length, 200
    chain->SetBranchAddress("E", &E);
    chain->SetBranchAddress("px", &px);
    chain->SetBranchAddress("py", &py);

    TH1D* hist = new TH1D(hist_name, title, 250, 0, 5000); // empirically determined maximum
    
    for (Long64_t i = 0; i < nentries; i++) {
        chain->GetEntry(i);
        if(sig==is_signal){
            for (Int_t j = 0; j < n; j++) {
                if(E[j] <= 0.) break;
                Double_t pT = TMath::Sqrt(px[j] * px[j] + py[j] * py[j]);
                hist->Fill(pT);
            }
        }
    }
    delete chain;
    return hist;
}

// Get a histogram of the average pT of jet constituents
TH1D* AvgConstituentPTHist(TString title, TString directory, TString hist_name, Int_t is_signal = 1){
    Int_t sig;
    Int_t n;
    Double_t E[200];
    Double_t px[200];
    Double_t py[200];
    TChain* chain = GetChain(directory);
    Long64_t nentries = chain->GetEntries();
    chain->SetBranchAddress("is_signal", &sig);
    chain->SetBranchAddress("n", &n); // this will always be the max length, 200
    chain->SetBranchAddress("E", &E);
    chain->SetBranchAddress("px", &px);
    chain->SetBranchAddress("py", &py);
    
    TH1D* hist = new TH1D(hist_name, title, 250, 0, 5000); // empirically determined maximum
    
    for (Long64_t i = 0; i < nentries; i++) {
        chain->GetEntry(i);
        if(sig==is_signal){
            Double_t avg_pt = 0.;
            Double_t avg_n = 0.;
            for (Int_t j = 0; j < n; j++) {
                if(E[j] <= 0.) break;
                avg_pt += TMath::Sqrt(px[j] * px[j] + py[j] * py[j]);
                avg_n += 1.;
            }
            avg_pt = avg_pt / avg_n;
            hist->Fill(avg_pt);
        }
    }
    delete chain;
    return hist;
}

// Get a histogram of the difference between maximum and minimum pT of jet constituents
TH1D* DiffConstituentPTHist(TString title, TString directory, TString hist_name, Int_t is_signal = 1){
    Int_t sig;
    Int_t n;
    Double_t E[200];
    Double_t px[200];
    Double_t py[200];
    Double_t pt_max;
    Double_t pt_min;
    TChain* chain = GetChain(directory);
    Long64_t nentries = chain->GetEntries();
    chain->SetBranchAddress("is_signal", &sig);
    chain->SetBranchAddress("n", &n); // this will always be the max length, 200
    chain->SetBranchAddress("E", &E);
    chain->SetBranchAddress("px", &px);
    chain->SetBranchAddress("py", &py);
    
    TH1D* hist = new TH1D(hist_name, title, 250, 0, 5000); // empirically determined maximum
    
    for (Long64_t i = 0; i < nentries; i++) {
        chain->GetEntry(i);
        if(sig==is_signal){
            pt_max = -1.;
            pt_min = -1.;
            for (Int_t j = 0; j < n; j++) {
                if(E[j] <= 0.) break;
                Double_t pT = TMath::Sqrt(px[j] * px[j] + py[j] * py[j]);
                
                if(pT < pt_min || pt_min < 0.) pt_min = pT;
                else if (pT > pt_max) pt_max = pT;
            }
            hist->Fill(pt_max - pt_min);
        }
    }
    delete chain;
    return hist;
}

// Get a histogram of the pt of jets
TH1D* PTHist(TString title, TString directory, TString hist_name, Int_t is_signal = 1){
    Int_t sig;
    Double_t jet_pt;
    TChain* chain = GetChain(directory);
    Long64_t nentries = chain->GetEntries();
    chain->SetBranchAddress("is_signal", &sig);
    chain->SetBranchAddress("jet_pt", &jet_pt);
    
    TH1D* hist = new TH1D(hist_name, title, 250, 0, 5000.); // empirically determined maximum
    
    for (Long64_t i = 0; i < nentries; i++) {
        chain->GetEntry(i);
        if(sig==is_signal) hist->Fill(jet_pt);
    }
    delete chain;
    return hist;
}

// Get a histogram of ∆E/∆m (calculated for every pair of jet constituents in a given event)
// Note that this is very time-consuming for large numbers of events!
TH1D* DeltaEDeltaMHist(TString title, TString directory, TString hist_name, Int_t is_signal = 1){
    Int_t sig;
    Int_t n;
    Double_t E[200];
    Double_t px[200];
    Double_t py[200];
    Double_t pz[200];
    TChain* chain = GetChain(directory);
    Long64_t nentries = chain->GetEntries();
    chain->SetBranchAddress("is_signal", &sig);
    chain->SetBranchAddress("n", &n); // this will always be the max length, 200
    chain->SetBranchAddress("E", &E);
    chain->SetBranchAddress("px", &px);
    chain->SetBranchAddress("py", &py);
    chain->SetBranchAddress("pz", &pz);
    
    TH1D* hist = new TH1D(hist_name, title, 1000, 0, 5000); // empirically determined maximum

    for (Long64_t i = 0; i < nentries; i++) {
        chain->GetEntry(i);
        if(sig==is_signal){
            for (Int_t j = 0; j < n; j++) {
                for (Int_t k = j+1; k < n; k++) {
                    Double_t deltaE = E[k] - E[j];
                    Double_t deltaM = TMath::Sqrt(E[k] * E[k] - (px[k] * px[k] + py[k] * py[k] + pz[k] * pz[k]));
                    deltaM -= TMath::Sqrt(E[j] * E[j] - (px[j] * px[j] + py[j] * py[j] + pz[j] * pz[j]));
                    hist->Fill(deltaE/deltaM);
                }
                
            }
        }
    }
    delete chain;
    return hist;
}

// Get a histogram of the difference between max and min E/m for each event
TH1D* DeltaEMDiffHist(TString title, TString directory, TString hist_name, Int_t is_signal = 1){
    Int_t sig;
    Int_t n;
    Double_t E[200];
    Double_t px[200];
    Double_t py[200];
    Double_t pz[200];
    TChain* chain = GetChain(directory);
    Long64_t nentries = chain->GetEntries();
    chain->SetBranchAddress("is_signal", &sig);
    chain->SetBranchAddress("n", &n); // this will always be the max length, 200
    chain->SetBranchAddress("E", &E);
    chain->SetBranchAddress("px", &px);
    chain->SetBranchAddress("py", &py);
    chain->SetBranchAddress("pz", &pz);
    TH1D* hist = new TH1D(hist_name, title, 100, 0., 100000000.); // empirically determined maximum
    
    for (Long64_t i = 0; i < nentries; i++) {
        chain->GetEntry(i);
        if(sig==is_signal){
            Double_t max_EM = -1.;
            Double_t min_EM = -1.;
            for (Int_t j = 0; j < n; j++) {
                Double_t m = TMath::Sqrt(E[j] * E[j] - (px[j] * px[j] + py[j] * py[j] + pz[j] * pz[j])); // sqrt(E^2 - p^2)
                Double_t EM = E[j]/m;
                
                if(EM > max_EM) max_EM = EM;
                else if (EM < min_EM || j == 0) min_EM = EM;
            }
            hist->Fill(max_EM - min_EM);
        }
    }
    delete chain;
    return hist;
}

// Get a histogram of the number of jet constituents.
TH1I* NConstituentsHist(TString title, TString directory, TString hist_name, Int_t is_signal = 1, Int_t nbins = 202){
    Int_t sig;
    Int_t n;
    Double_t E[200];
    TChain* chain = GetChain(directory);
    Long64_t nentries = chain->GetEntries();
    chain->SetBranchAddress("is_signal", &sig);
    chain->SetBranchAddress("n", &n); // this will always be the max length, 200
    chain->SetBranchAddress("E", &E);
    
    TH1I* hist = new TH1I(hist_name, title, nbins, 0, nbins);
    
    for (Long64_t i = 0; i < nentries; i++) {
        chain->GetEntry(i);
        if(sig==is_signal){
            Int_t n_actual = 0;
            for (Int_t j = 0; j < n; j++) {
                if(E[j] <= 0.) break;
                n_actual++;
            }
            hist->Fill(n_actual);
        }
    }
    delete chain;
    return hist;
}

void JetHistograms(Int_t run_number, Bool_t MPI = kTRUE, Bool_t noMPI = kTRUE){
    TString run_folder = TString("run").Append(std::to_string(run_number));
    TString dir1 = "../samples_root/reduced/XXX/parton/noMPI";
    TString dir2 = "../samples_root/reduced/XXX/parton/MPI";
    dir1.ReplaceAll("XXX", run_folder);
    dir2.ReplaceAll("XXX", run_folder);
    
    TString title1 = "Jet Mass, XXX;m(jet) (GeV);Events";
    TString title2 = "Jet p_{T}, XXX;p_{T}(jet) (GeV);Events";
    TString title3 = "N(constituents), XXX;N;Events";
    title1.ReplaceAll("XXX", run_folder);
    title2.ReplaceAll("XXX", run_folder);
    title3.ReplaceAll("XXX", run_folder);

    TCanvas* c1 = new TCanvas("canv1", "canv1", 800, 600);
    TCanvas* c2 = new TCanvas("canv2", "canv2", 800, 600);
    TCanvas* c3 = new TCanvas("canv3", "canv3", 800, 600);
    gStyle->SetOptStat(0);
    
    // mass histograms
    TH1D* mass_sig_nMPI = new TH1D("h1s_m", "h1s_m",2,0,1);
    TH1D* mass_bck_nMPI = new TH1D("h1b_m", "h1b_m",2,0,1);
    TH1D* mass_sig_yMPI = new TH1D("h2s_m", "h2s_m",2,0,1);
    TH1D* mass_bck_yMPI = new TH1D("h2b_m", "h2b_m",2,0,1);
    mass_sig_nMPI->SetLineColor(kRed);
    mass_bck_nMPI->SetLineColor(kOrange);
    mass_sig_yMPI->SetLineColor(kBlue);
    mass_bck_yMPI->SetLineColor(kViolet);
    THStack* mass_stack = new THStack("stack1", title1);
    
    // pt histograms
    TH1D* pt_sig_nMPI = new TH1D("h1s_pt", "h1s_pt",2,0,1);
    TH1D* pt_bck_nMPI = new TH1D("h1b_pt", "h1b_pt",2,0,1);
    TH1D* pt_sig_yMPI = new TH1D("h2s_pt", "h2s_pt",2,0,1);
    TH1D* pt_bck_yMPI = new TH1D("h2b_pt", "h2b_pt",2,0,1);
    pt_sig_nMPI->SetLineColor(kRed);
    pt_bck_nMPI->SetLineColor(kOrange);
    pt_sig_yMPI->SetLineColor(kBlue);
    pt_bck_yMPI->SetLineColor(kViolet);
    THStack* pt_stack = new THStack("stack2", title2);
    
    // n_const histograms
    TH1I* n_sig_nMPI = new TH1I("h1s_n", "h1s_n",2,0,1);
    TH1I* n_bck_nMPI = new TH1I("h1b_n", "h1b_n",2,0,1);
    TH1I* n_sig_yMPI = new TH1I("h2s_n", "h2s_n",2,0,1);
    TH1I* n_bck_yMPI = new TH1I("h2b_n", "h2b_n",2,0,1);
    n_sig_nMPI->SetLineColor(kRed);
    n_bck_nMPI->SetLineColor(kOrange);
    n_sig_yMPI->SetLineColor(kBlue);
    n_bck_yMPI->SetLineColor(kViolet);
    THStack* n_stack = new THStack("stack3", title3);
    
    
    if(noMPI){
        mass_sig_nMPI = MassHist(title1, dir1, "h1s_m", 1);
        mass_bck_nMPI = MassHist(title1, dir1, "h1b_m", 0);
        mass_sig_nMPI->SetLineColor(kRed);
        mass_bck_nMPI->SetLineColor(kOrange);
        mass_stack->Add(mass_sig_nMPI);
        mass_stack->Add(mass_bck_nMPI);
        
        pt_sig_nMPI = PTHist(title2, dir1, "h1s_pt", 1);
        pt_bck_nMPI = PTHist(title2, dir1, "h1b_pt", 0);
        pt_sig_nMPI->SetLineColor(kRed);
        pt_bck_nMPI->SetLineColor(kOrange);
        pt_stack->Add(pt_sig_nMPI);
        pt_stack->Add(pt_bck_nMPI);
        
        n_sig_nMPI = NConstituentsHist(title3,dir1, "h1s_n", 1, 30);
        n_bck_nMPI = NConstituentsHist(title3, dir1, "h1b_n", 0, 30);
        n_sig_nMPI->SetLineColor(kRed);
        n_bck_nMPI->SetLineColor(kOrange);
        n_stack->Add(n_sig_nMPI);
        n_stack->Add(n_bck_nMPI);
        
    }
    if(MPI){
        mass_sig_yMPI = MassHist(title1, dir2, "h2s_m", 1);
        mass_bck_yMPI = MassHist(title1, dir2, "h2b_m", 0);
        mass_sig_yMPI->SetLineColor(kBlue);
        mass_bck_yMPI->SetLineColor(kViolet);
        mass_stack->Add(mass_sig_yMPI);
        mass_stack->Add(mass_bck_yMPI);
        
        pt_sig_yMPI = PTHist(title2, dir2, "h2s_pt", 1);
        pt_bck_yMPI = PTHist(title2, dir2, "h2b_pt", 0);
        pt_sig_yMPI->SetLineColor(kBlue);
        pt_bck_yMPI->SetLineColor(kViolet);
        pt_stack->Add(pt_sig_yMPI);
        pt_stack->Add(pt_bck_yMPI);
        
        n_sig_yMPI = NConstituentsHist(title3, dir2, "h2s_n", 1, 30);
        n_bck_yMPI = NConstituentsHist(title3, dir2, "h2b_n", 0, 30);
        n_sig_yMPI->SetLineColor(kBlue);
        n_bck_yMPI->SetLineColor(kViolet);
        n_stack->Add(n_sig_yMPI);
        n_stack->Add(n_bck_yMPI);
    }
    
    TLegend* leg = SetupLegend(0.65, 0.7, 0.9,0.85);
    if (noMPI){
        leg->AddEntry(mass_sig_nMPI, "t#bar{t}, no MPI", "l");
        leg->AddEntry(mass_bck_nMPI, "QCD, no MPI", "l");
    }
    if(MPI){
        leg->AddEntry(mass_sig_yMPI, "t#bar{t}, MPI", "l");
        leg->AddEntry(mass_bck_yMPI, "QCD, MPI", "l");
    }
    
    c1->cd();
    c1->SetLogy();
    mass_stack->Draw("nostack");
    leg->Draw();
    c1->Draw();
    c1->SaveAs("jet_m.eps");
    c1->SaveAs("jet_m.png");
    
    c2->cd();
    c2->SetLogy();
    pt_stack->Draw("nostack");
    leg->Draw();
    c2->Draw();
    c2->SaveAs("jet_pt.eps");
    c2->SaveAs("jet_pt.png");
    
    c3->cd();
    n_stack->Draw("nostack");
    leg->Draw();
    c3->Draw();
    c3->SaveAs("jet_nconst.eps");
    c3->SaveAs("jet_nconst.png");
    
    return;
}

// For a given run, make histograms for various properties of jet constituents
void ConstPlots(Int_t run_number, Bool_t MPI = kTRUE, Bool_t noMPI = kTRUE){
    
    Int_t options [5] = {0, 0, 1, 0, 1}; // n, E, pt, E/m, m
    
    TString run_folder = TString("run").Append(std::to_string(run_number));
    TString dir1 = "../samples_root/reduced/XXX/parton/MPI";
    TString dir2 = "../samples_root/reduced/XXX/parton/noMPI";
    dir1.ReplaceAll("XXX", run_folder);
    dir2.ReplaceAll("XXX", run_folder);

    TString title1 = "Number of Jet Constituents;N(const);Events";
    TString title2a = "Jet Constituent Energy;E(const) (GeV);Events";
    TString title2b = "Avg. Jet Constituent Energy;#bar{E}(const) (GeV);Events";
    TString title2c = "Max-Min Jet Constituent Energy;#Delta E(const) (GeV);Events";
    TString title3a = "Jet Constituent pT;pT(const) (GeV);Events";
    TString title3b = "Avg. Jet Constituent pT;#bar{pT}(const) (GeV);Events";
    TString title3c = "Max-Min Jet Constituent pT;#Delta pT(const) (GeV);Events";
    TString title4a = "#Delta E / #Delta m;#Delta E / #Delta m ;Events";
    TString title4b = "Max-Min E / m;#Delta (E / m) (GeV);Events";
    TString title5 = "Jet Constituent Mass;M(const) (GeV);Events";

    TCanvas* c1 = new TCanvas("c1", "c1", 800, 600);
    TCanvas* c2a = new TCanvas("c2a", "c2a", 800, 600);
    TCanvas* c2b = new TCanvas("c2b", "c2b", 800, 600);
    TCanvas* c2c = new TCanvas("c2c", "c2c", 800, 600);
    TCanvas* c3a = new TCanvas("c3a", "c3a", 800, 600);
    TCanvas* c3b = new TCanvas("c3b", "c3b", 800, 600);
    TCanvas* c3c = new TCanvas("c3c", "c3c", 800, 600);
    TCanvas* c4a = new TCanvas("c4a", "c4a", 800, 600);
    TCanvas* c4b = new TCanvas("c4b", "c4b", 800, 600);
    TCanvas* c5 = new TCanvas("c5", "c5", 800, 600);
    gStyle->SetOptStat(0);
    
    // n-constituent hists
    TH1I* hist1s_n;
    TH1I* hist1b_n;
    TH1I* hist2s_n;
    TH1I* hist2b_n;
    
    // E hists
    TH1D* hist1s_E;
    TH1D* hist1b_E;
    TH1D* hist2s_E;
    TH1D* hist2b_E;
    
    // avg. E hists
    TH1D* hist1s_E_avg;
    TH1D* hist1b_E_avg;
    TH1D* hist2s_E_avg;
    TH1D* hist2b_E_avg;
    
    // diff. E hists
    TH1D* hist1s_E_diff;
    TH1D* hist1b_E_diff;
    TH1D* hist2s_E_diff;
    TH1D* hist2b_E_diff;
    
    // pT hists
    TH1D* hist1s_pt;
    TH1D* hist1b_pt;
    TH1D* hist2s_pt;
    TH1D* hist2b_pt;
    
    // avg. pT hists
    TH1D* hist1s_pt_avg;
    TH1D* hist1b_pt_avg;
    TH1D* hist2s_pt_avg;
    TH1D* hist2b_pt_avg;
    
    // diff. pT hists
    TH1D* hist1s_pt_diff;
    TH1D* hist1b_pt_diff;
    TH1D* hist2s_pt_diff;
    TH1D* hist2b_pt_diff;
    
    // E/m hists
    TH1D* hist1s_Em;
    TH1D* hist1b_Em;
    TH1D* hist2s_Em;
    TH1D* hist2b_Em;
    
    // (max-min) E/m hists
    TH1D* hist1s_Em_diff;
    TH1D* hist1b_Em_diff;
    TH1D* hist2s_Em_diff;
    TH1D* hist2b_Em_diff;
    
    // mass hists
    TH1D* hist1s_m;
    TH1D* hist1b_m;
    TH1D* hist2s_m;
    TH1D* hist2b_m;
    
    THStack* hists_n = new THStack("stack1", title1);
    THStack* hists_E = new THStack("stack2a", title2a);
    THStack* hists_pt = new THStack("stack3a", title3a);
    THStack* hists_E_avg = new THStack("stack2b", title2b);
    THStack* hists_pt_avg = new THStack("stack3b", title3b);
    THStack* hists_E_diff = new THStack("stack2c", title2c);
    THStack* hists_pt_diff = new THStack("stack3c", title3c);
    THStack* hists_Em = new THStack("stack4a", title4a);
    THStack* hists_Em_diff = new THStack("stack4b", title4b);
    THStack* hists_m = new THStack("stack5", title5);

    if(MPI){
        
        if(options[0] == 1){
            hist1s_n = NConstituentsHist(title1,dir1, "h1s_n", 1);
            hist1b_n = NConstituentsHist(title1,dir1, "h1b_n", 0);
            hist1s_n->SetLineColor(kBlue);
            hist1b_n->SetLineColor(kViolet);
            hists_n->Add(hist1s_n);
            hists_n->Add(hist1b_n);
        }
        // using these hists for legend, so they must exist somehow
        else{
            hist1s_n = new TH1I("h1s_n", "h1s_n",2,0,1);
            hist1b_n = new TH1I("h1b_n", "h1b_n",2,0.,1.);
            hist1s_n->SetLineColor(kBlue);
            hist1b_n->SetLineColor(kViolet);
        }
        
        if(options[1] == 1){
            hist1s_E = ConstituentEnergyHist(title2a,dir1, "h1s_E", 1);
            hist1b_E = ConstituentEnergyHist(title2a,dir1, "h1b_E", 0);
            hist1s_E->SetLineColor(kBlue);
            hist1b_E->SetLineColor(kViolet);
            hists_E->Add(hist1s_E);
            hists_E->Add(hist1b_E);
        
            hist1s_E_avg = AvgConstituentEnergyHist(title2b,dir1, "h1s_E_avg", 1);
            hist1b_E_avg = AvgConstituentEnergyHist(title2b,dir1, "h1b_E_avg", 0);
            hist1s_E_avg->SetLineColor(kBlue);
            hist1b_E_avg->SetLineColor(kViolet);
            hists_E_avg->Add(hist1s_E_avg);
            hists_E_avg->Add(hist1b_E_avg);
        
            hist1s_E_diff = DiffConstituentEnergyHist(title2c,dir1, "h1s_E_diff", 1);
            hist1b_E_diff = DiffConstituentEnergyHist(title2c,dir1, "h1b_E_diff", 0);
            hist1s_E_diff->SetLineColor(kBlue);
            hist1b_E_diff->SetLineColor(kViolet);
            hists_E_diff->Add(hist1s_E_diff);
            hists_E_diff->Add(hist1b_E_diff);
        }
        
        if(options[2] == 1){
            hist1s_pt = ConstituentPTHist(title3a,dir1, "h1s_pt", 1);
            hist1b_pt = ConstituentPTHist(title3a,dir1, "h1b_pt", 0);
            hist1s_pt->SetLineColor(kBlue);
            hist1b_pt->SetLineColor(kViolet);
            hists_pt->Add(hist1s_pt);
            hists_pt->Add(hist1b_pt);
        
            hist1s_pt_avg = AvgConstituentPTHist(title3b,dir1, "h1s_pt_avg", 1);
            hist1b_pt_avg = AvgConstituentPTHist(title3b,dir1, "h1b_pt_avg", 0);
            hist1s_pt_avg->SetLineColor(kBlue);
            hist1b_pt_avg->SetLineColor(kViolet);
            hists_pt_avg->Add(hist1s_pt_avg);
            hists_pt_avg->Add(hist1b_pt_avg);
        
            hist1s_pt_diff = DiffConstituentPTHist(title3c,dir1, "h1s_pt_diff", 1);
            hist1b_pt_diff = DiffConstituentPTHist(title3c,dir1, "h1b_pt_diff", 0);
            hist1s_pt_diff->SetLineColor(kBlue);
            hist1b_pt_diff->SetLineColor(kViolet);
            hists_pt_diff->Add(hist1s_pt_diff);
            hists_pt_diff->Add(hist1b_pt_diff);
        }
        
//        hist1s_Em = DeltaEDeltaMHist(title4a,dir1, "h1s_Em", 1);
//        hist1b_Em = DeltaEDeltaMHist(title4a,dir1, "h1b_Em", 0);
//        hist1s_Em->SetLineColor(kBlue);
//        hist1b_Em->SetLineColor(kViolet);
//        hists_Em->Add(hist1s_Em);
//        hists_Em->Add(hist1b_Em);
        
        if(options[3] == 1){
            hist1s_Em_diff = DeltaEMDiffHist(title4b,dir1, "h1s_Em_diff", 1);
            hist1b_Em_diff = DeltaEMDiffHist(title4b,dir1, "h1b_Em_diff", 0);
            hist1s_Em_diff->SetLineColor(kBlue);
            hist1b_Em_diff->SetLineColor(kViolet);
            hists_Em_diff->Add(hist1s_Em_diff);
            hists_Em_diff->Add(hist1b_Em_diff);
        }
        
        if(options[4] == 1){
            hist1s_m = ConstituentMassHist(title5,dir1, "h1s_m", 1);
            hist1b_m = ConstituentMassHist(title5,dir1, "h1b_m", 0);
            hist1s_m->SetLineColor(kBlue);
            hist1b_m->SetLineColor(kViolet);
            hists_m->Add(hist1s_m);
            hists_m->Add(hist1b_m);
        }
    }
    
    if(noMPI){
        if(options[0] == 1){
            hist2s_n = NConstituentsHist(title1,dir2, "h2s_n", 1);
            hist2b_n = NConstituentsHist(title1,dir2, "h2b_n", 0);
            hist2s_n->SetLineColor(kRed);
            hist2b_n->SetLineColor(kOrange);
            hists_n->Add(hist2s_n);
            hists_n->Add(hist2b_n);
        }
        // using these hists for legend, so they must exist somehow
        else{
            hist2s_n = new TH1I("h2s_n", "h2s_n",2,0,1);
            hist2b_n = new TH1I("h2b_n", "h2b_n",2,0,1);
            hist2s_n->SetLineColor(kRed);
            hist2b_n->SetLineColor(kOrange);
        }
        
        if(options[1] == 1){
            hist2s_E = ConstituentEnergyHist(title2a,dir2, "h2s_E", 1);
            hist2b_E = ConstituentEnergyHist(title2a,dir2, "h2b_E", 0);
            hist2s_E->SetLineColor(kRed);
            hist2b_E->SetLineColor(kOrange);
            hists_E->Add(hist2s_E);
            hists_E->Add(hist2b_E);
        
            hist2s_E_avg = AvgConstituentEnergyHist(title2b,dir2, "h2s_E_avg", 1);
            hist2b_E_avg = AvgConstituentEnergyHist(title2b,dir2, "h2b_E_avg", 0);
            hist2s_E_avg->SetLineColor(kRed);
            hist2b_E_avg->SetLineColor(kOrange);
            hists_E_avg->Add(hist2s_E_avg);
            hists_E_avg->Add(hist2b_E_avg);
        
            hist2s_E_diff = DiffConstituentEnergyHist(title2c,dir2, "h2s_E_diff", 1);
            hist2b_E_diff = DiffConstituentEnergyHist(title2c,dir2, "h2b_E_diff", 0);
            hist2s_E_diff->SetLineColor(kRed);
            hist2b_E_diff->SetLineColor(kOrange);
            hists_E_diff->Add(hist2s_E_diff);
            hists_E_diff->Add(hist2b_E_diff);
        }
        
        if(options[2] == 1){
            hist2s_pt = ConstituentPTHist(title3a,dir2, "h2s_pt", 1);
            hist2b_pt = ConstituentPTHist(title3a,dir2, "h2b_pt", 0);
            hist2s_pt->SetLineColor(kRed);
            hist2b_pt->SetLineColor(kOrange);
            hists_pt->Add(hist2s_pt);
            hists_pt->Add(hist2b_pt);
        
            hist2s_pt_avg = AvgConstituentPTHist(title3b,dir2, "h2s_pt_avg", 1);
            hist2b_pt_avg = AvgConstituentPTHist(title3b,dir2, "h2b_pt_avg", 0);
            hist2s_pt_avg->SetLineColor(kRed);
            hist2b_pt_avg->SetLineColor(kOrange);
            hists_pt_avg->Add(hist2s_pt_avg);
            hists_pt_avg->Add(hist2b_pt_avg);
        
            hist2s_pt_diff = DiffConstituentPTHist(title3c,dir2, "h2s_pt_diff", 1);
            hist2b_pt_diff = DiffConstituentPTHist(title3c,dir2, "h2b_pt_diff", 0);
            hist2s_pt_diff->SetLineColor(kRed);
            hist2b_pt_diff->SetLineColor(kOrange);
            hists_pt_diff->Add(hist2s_pt_diff);
            hists_pt_diff->Add(hist2b_pt_diff);
        }
        
//        hist2s_Em = DeltaEDeltaMHist(title4a,dir2, "h2s_Em", 1);
//        hist2b_Em = DeltaEDeltaMHist(title4a,dir2, "h2b_Em", 0);
//        hist2s_Em->SetLineColor(kRed);
//        hist2b_Em->SetLineColor(kOrange);
//        hists_Em->Add(hist2s_Em);
//        hists_Em->Add(hist2b_Em);
        
        if(options[3] == 1){
            hist2s_Em_diff = DeltaEMDiffHist(title4b,dir2, "h2s_Em_diff", 1);
            hist2b_Em_diff = DeltaEMDiffHist(title4b,dir2, "h2b_Em_diff", 0);
            hist2s_Em_diff->SetLineColor(kRed);
            hist2b_Em_diff->SetLineColor(kOrange);
            hists_Em_diff->Add(hist2s_Em_diff);
            hists_Em_diff->Add(hist2b_Em_diff);
        }
        
        if(options[4] == 1){
            hist2s_m = ConstituentMassHist(title5,dir2, "h2s_m", 1);
            hist2b_m = ConstituentMassHist(title5,dir2, "h2b_m", 0);
            hist2s_m->SetLineColor(kRed);
            hist2b_m->SetLineColor(kOrange);
            hists_m->Add(hist2s_m);
            hists_m->Add(hist2b_m);
        }
    }

    TLegend* leg = SetupLegend(0.65, 0.7, 0.9,0.85);
    if(MPI){
        leg->AddEntry(hist1s_n, "t#bar{t}, MPI", "l");
        leg->AddEntry(hist1b_n, "QCD, MPI", "l");
    }
    if (noMPI){
        leg->AddEntry(hist2s_n, "t#bar{t}, no MPI", "l");
        leg->AddEntry(hist2b_n, "QCD, no MPI", "l");
    }
    
    c1->cd();
    hists_n->Draw("nostack");
    leg->Draw();
    
    c2a->cd();
    c2a->SetLogy();
    hists_E->Draw("nostack");
    leg->Draw();
    
    c2b->cd();
    c2b->SetLogy();
    hists_E_avg->Draw("nostack");
    leg->Draw();
    
    c2c->cd();
    c2c->SetLogy();
    hists_E_diff->Draw("nostack");
    leg->Draw();
    
    c3a->cd();
    c3a->SetLogy();
    hists_pt->Draw("nostack");
    leg->Draw();
    
    c3b->cd();
    c3b->SetLogy();
    hists_pt_avg->Draw("nostack");
    leg->Draw();
    
    c3c->cd();
    c3c->SetLogy();
    hists_pt_diff->Draw("nostack");
    leg->Draw();
    
//    c4a->cd();
//    c4a->SetLogy();
//    hists_Em->Draw("nostack");
//    leg->Draw();
    
    c4b->cd();
//    c4b->SetLogx();
    c4b->SetLogy();
    hists_Em_diff->Draw("nostack");
    leg->Draw();
    
    c5->cd();
    c5->SetLogy();
    hists_m->Draw("nostack");
    leg->Draw();
    
    c1->Draw();
    c2a->Draw();
    c2b->Draw();
    c2c->Draw();

    c3a->Draw();
    c3b->Draw();
    c3c->Draw();
//    c4a->Draw();
    c4b->Draw();
    c5->Draw();
    
    c1->SaveAs("n_const.eps");
    c2a->SaveAs("E_const.eps");
    c2b->SaveAs("E_avg.eps");
    c2c->SaveAs("E_diff.eps");
    c3a->SaveAs("pt_const.eps");
    c3b->SaveAs("pt_avg.eps");
    c3c->SaveAs("pt_diff.eps");
//    c4a->SaveAs("Em.eps");
    c4b->SaveAs("Em_diff.eps");
    c5->SaveAs("m.eps");
    
    c1->SaveAs("n_const.png");
    c2a->SaveAs("E_const.png");
    c2b->SaveAs("E_avg.png");
    c2c->SaveAs("E_diff.png");
    c3a->SaveAs("pt_const.png");
    c3b->SaveAs("pt_avg.png");
    c3c->SaveAs("pt_diff.png");
    c4b->SaveAs("Em_diff.png");
    c5->SaveAs("m.png");

    return;
}

void JetPlotBinned(Int_t nconst, Double_t* eta, Double_t* phi, Double_t* pt, Int_t sig = 0, Float_t offset = 1.){
    
    // make a Lego plot of the jet constituents, do some formatting
    Double_t x_min = -3.15;
    Double_t x_max = -1. * x_min;
    Double_t y_min = 0.;
    Double_t y_max = x_max - x_min;
    Int_t nbins = (Int_t)(y_max / 0.1);

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
    if(sig == 0) process.Append("QCD");
    else process.Append("t#bar{t}");
    pave->AddText(TString("process = ").Append(process));

    pave->Draw();
    gStyle->SetOptStat(0);
    canv->Draw();
    return;
}

void JetPlotUnbinned(Int_t nconst, Double_t* eta, Double_t* phi, Double_t jeta, Double_t jphi, Double_t jet_radius, Double_t eta_max, Int_t sig = 0, Double_t tphi = 0., Double_t teta = 0.){
    
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

void JetPlot(TString dir, Long64_t event_index, Double_t eta_max = 2.0, Int_t mode = 0, Float_t offset = 1.5){
    //mode = 0: unbinned plot
    //mode = 1: binned plot
    
    Double_t jet_radius = 0.8;
    TChain* chain = GetChain(dir);
    Long64_t nentries = chain->GetEntries();
    if(event_index >= nentries){
        std::cout << "Warning: event_index > nentries = " << nentries << std::endl;
        return;
    }
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
    // setting branch addresses for the variables above
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

    // select the entry # as given by the user
    chain->GetEntry(event_index);
    
    // explicitly count the # of non-zero jet constituents
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
    else{
        JetPlotBinned(nconst, eta, phi, pt, sig, offset);
    }
    
    
    return;
}

