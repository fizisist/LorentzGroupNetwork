//  present.C
//  Created by Jan Offermann on 01/04/20.
#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <functional>

#include "TSystemDirectory.h"
#include "TStyle.h"
#include "TString.h"
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TList.h"
#include "TObject.h"
#include "TObjString.h"
#include "TParameter.h"
#include "TLorentzVector.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "TH2D.h"
#include "THStack.h"
#include "TLegend.h"
#include "TPaveText.h"
#include "TMath.h"

// Helper function for making TPaveText
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

TLegend* SetupLegend(Double_t x1, Double_t y1, Double_t x2, Double_t y2){
    TLegend* leg = new TLegend(x1,y1,x2,y2);
    leg->SetFillColor(0);
    leg->SetFillStyle(0); // make it transparent!
    leg->SetBorderSize(0);
    leg->SetTextFont(42);
    leg->SetTextAngle(0);
    leg->SetTextColor(kBlack);
    leg->SetTextSize(0.04);
    leg->SetTextAlign(12);
    return leg;
}

void PlotAdjust(TH1* histogram, Double_t x_min, Double_t x_max, Double_t y_min, Double_t y_max, TString title = "", Bool_t scientific = kFALSE){
    Double_t label_size = 0.05;
    Double_t title_size = 0.05;
    Double_t title_offset_x = 0.875;
    Double_t title_offset_y = 0.9;
    Double_t line_thickness_multiplier = 2.5;
    
    histogram->GetXaxis()->SetLabelSize(label_size);
    histogram->GetYaxis()->SetLabelSize(label_size);
    histogram->GetXaxis()->SetTitleSize(title_size);
    histogram->GetYaxis()->SetTitleSize(title_size);
    histogram->GetXaxis()->SetTitleOffset(title_offset_x);
    histogram->GetYaxis()->SetTitleOffset(title_offset_y);
    histogram->SetLineWidth(histogram->GetLineWidth() * line_thickness_multiplier);
    histogram->GetXaxis()->SetRangeUser(x_min, x_max);
    histogram->GetYaxis()->SetRangeUser(y_min, y_max);
    if(!title.EqualTo("")) histogram->SetTitle(title);
    if(scientific){
//        std::cout << "ha" << std::endl;
//        histogram->GetXaxis()->SetMaxDigits(2);
        histogram->GetYaxis()->SetMaxDigits(3);
//        TGaxis::SetMaxDigits(2);
    }
    
    return;
}

void PlotDisplay(TCanvas* c, TH1* sig, TH1* bck, TLegend* leg, TPaveText* pave, Bool_t log = kTRUE){
    
    c->cd();
    sig->Draw();
    bck->Draw("SAME");
    leg->Draw();
    pave->Draw();
    if(log) c->SetLogy();
    c->Draw();
    c->Update();
    return;
}

void CanvasSave(TCanvas* c, TString prefix, TFile* f){
    TString name1 = TString(prefix).Append(".eps");
    TString name2 = TString(prefix).Append(".png");
    c->SaveAs(name1);
    c->SaveAs(name2);
    f->cd();
    c->Write("",TObject::kOverwrite);
    return;
}

void MakePlots(TString filename = "plots_h5.root"){
    
    // Setup ATLAS Style
    gROOT->SetStyle("ATLAS"); // works since ROOT version 6.13
    gROOT->ForceStyle();
    
    // Some style adjustments
    gStyle->SetHistLineWidth(1); // no bold lines, these are too thick
    
    TFile* f = new TFile(filename,"READ");
    // Jet histograms
    TH1F* pt_j_sig = (TH1F*)f->Get("pt_j_hist_sig");
    TH1F* pt_j_bck = (TH1F*)f->Get("pt_j_hist_bck");
    pt_j_sig->SetLineColor(kBlue);
    pt_j_bck->SetLineColor(kRed);
    
    TH1F* eta_j_sig = (TH1F*)f->Get("eta_j_hist_sig");
    TH1F* eta_j_bck = (TH1F*)f->Get("eta_j_hist_bck");
    eta_j_sig->SetLineColor(kBlue);
    eta_j_bck->SetLineColor(kRed);

    TH1F* phi_j_sig = (TH1F*)f->Get("phi_j_hist_sig");
    TH1F* phi_j_bck = (TH1F*)f->Get("phi_j_hist_bck");
    phi_j_sig->SetLineColor(kBlue);
    phi_j_bck->SetLineColor(kRed);
    // --------------------
    
    // Event histograms
    TH1F* nobj_sig = (TH1F*)f->Get("n_hist_sig");
    TH1F* nobj_bck = (TH1F*)f->Get("n_hist_bck");
    nobj_sig->SetLineColor(kBlue);
    nobj_bck->SetLineColor(kRed);
    // --------------------
    
    TLegend* leg = SetupLegend(0.5,0.725,0.9,0.9); // legend for use by all histograms
    leg->SetHeader("anti-k_{T}, R = 0.8");
    leg->AddEntry(pt_j_sig,"Hadronic top decays","l");
    leg->AddEntry(pt_j_bck,"Light quarks and gluons","l");
    
    TPaveText* pave = SetupPave(0.4, 1. - gStyle->GetPadTopMargin(), 1. - gStyle->GetPadRightMargin(),1.);
    pave->SetTextColor(kGray + 2);
    pave->SetTextFont(12); // times-medium-i-normal w/ precision = 2 (scalable & rotatable hardware font)
    pave->SetTextSize(0.04);
    pave->AddText("Top tagging reference dataset (arXiv:1902.09914)");
    
    // --- Some beautification of plots for display ---
    Double_t pt_min = 400.; // GeV
    Double_t pt_max = 800.;
    
    Double_t eta_min = -2.5;
    Double_t eta_max = -1. * eta_min;
    
    Double_t phi_min = -4.;
    Double_t phi_max = -1. * phi_min;
    
    Double_t n_min = 0.;
    Double_t n_max = 202.;
    
    Double_t pt_y_min = TMath::Power(10,4);
    Double_t pt_y_max = TMath::Power(10,6);
    
    Double_t eta_y_min = TMath::Power(10,3);
    Double_t eta_y_max = TMath::Power(10,5);
    
    Double_t phi_y_min = TMath::Power(10,3);
    Double_t phi_y_max = TMath::Power(10,5);
    
    Double_t n_y_min = 0.;
    Double_t n_y_max = 5. * TMath::Power(10,4);
    
    PlotAdjust((TH1*)pt_j_sig, pt_min, pt_max, pt_y_min, pt_y_max,";pT [GeV];Number of jets per 10 GeV");
    PlotAdjust((TH1*)pt_j_bck, pt_min, pt_max, pt_y_min, pt_y_max,";pT [GeV];Number of jets per 10 GeV");
    
    PlotAdjust((TH1*)eta_j_sig, eta_min, eta_max, eta_y_min, eta_y_max,";#eta;Number of jets per bin");
    PlotAdjust((TH1*)eta_j_bck, eta_min, eta_max, eta_y_min, eta_y_max,";#eta;Number of jets per bin");
    
    PlotAdjust((TH1*)phi_j_sig, phi_min, phi_max, phi_y_min, phi_y_max,";#phi;Number of jets per bin");
    PlotAdjust((TH1*)phi_j_bck, phi_min, phi_max, phi_y_min, phi_y_max,";#phi;Number of jets per bin");
    
    PlotAdjust((TH1*)nobj_sig, n_min, n_max, n_y_min, n_y_max,";N(particles);Number of events per bin",kTRUE);
    PlotAdjust((TH1*)nobj_bck, n_min, n_max, n_y_min, n_y_max,";N(particles);Number of events per bin",kTRUE);
   
    // Draw plots
    gStyle->SetOptStat(0);
    TCanvas* c_pt = new TCanvas("c1","c1",800,600);
    TCanvas* c_eta = new TCanvas("c2","c2",800,600);
    TCanvas* c_phi = new TCanvas("c3","c3",800,600);
    TCanvas* c_n = new TCanvas("c4","c4",800,600);
    
    PlotDisplay(c_pt, (TH1*)pt_j_sig, (TH1*)pt_j_bck, leg, pave);
    PlotDisplay(c_eta, (TH1*)eta_j_sig, (TH1*)eta_j_bck, leg, pave);
    PlotDisplay(c_phi, (TH1*)phi_j_sig, (TH1*)phi_j_bck, leg, pave);
    PlotDisplay(c_n, (TH1*)nobj_sig, (TH1*)nobj_bck, leg, pave, kFALSE);
    
    TFile* g = new TFile("plots.root", "UPDATE");
    CanvasSave(c_pt, "pt", g);
    CanvasSave(c_eta, "eta", g);
    CanvasSave(c_phi, "phi", g);
    CanvasSave(c_n, "n", g);
    g->Close();
    
    f->Close();
    return;
    
}
