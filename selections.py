from collections import OrderedDict

class Selections(object):

    def __init__(self, channel):
        self.channel = channel
        self.base    = None

        if user == 'manzoni': 
             env['BASE_DIR'] = '/Users/manzoni/Documents/efficiencyNN/HNL/mmm/ntuples/'
             env['PLOT_DIR'] = '/Users/manzoni/Documents/efficiencyNN/HNL/mmm/plots/'
             env['NN_DIR']   = '/Users/manzoni/Documents/efficiencyNN/HNL/mmm/NN/'

        if user == 'cesareborgia': 
            env['BASE_DIR'] = '/Users/cesareborgia/cernbox/ntuples/2018/'
            env['BASE_DIR'] = '/Users/cesareborgia/cernbox/ntuples/2018/'

        if self.channel == 'mmm':
            self.base = ' & '.join(['l0_pt > 25'    ,
                                    'l2_pt > 5'     ,
                                    'l1_pt > 5'     ,
                                    'l0_id_m == 1'  ,
                                    'l1_Medium == 1',
                                    'l2_Medium == 1',])

        if self.channel == 'mem':
            self.base = ' & '.join(['l0_pt > 25'         ,
                                    'l2_pt > 5'          ,
                                    'l1_pt > 5'          ,
                                    'l0_id_m == 1'       ,
                                    'l1_MediumNoIso == 1',
                                    'l2_Medium == 1'     ,])

        if self.channel == 'eem':
            self.base = ' & '.join(['l0_pt > 25'               ,
                                    'l2_pt > 5'                ,
                                    'l1_pt > 5'                ,
                                    'l0_eid_mva_iso_wp90 == 1' ,
                                    'l1_MediumNoIso == 1'      ,
                                    'l2_Medium == 1'           ,])

        if self.channel == 'eee':
            self.base = ' & '.join(['l0_pt > 25'               ,
                                    'l2_pt > 5'                ,
                                    'l1_pt > 5'                ,
                                    'l0_eid_mva_iso_wp90 == 1' ,
                                    'l1_MediumNoIso == 1'      ,
                                    'l2_MediumNoIso == 1'      ,])

        assert self.base, 'Error: No channel specific selection applied!'

        self.selections = OrderedDict()

        self.selections['baseline'] = ' & '.join([
            self.base                                  , 
            'abs(l0_eta) < 2.4'                        ,
            'abs(l0_dxy) < 0.05'                       ,
            'abs(l0_dz) < 0.2'                         ,
            'l0_reliso_rho_03 < 0.2'                   ,

            'abs(l1_eta) < 2.4'                        ,
#     'abs(l1_dxy) > 0.01'                       ,
            'l1_reliso_rho_03 < 10'                    ,

            'abs(l2_eta) < 2.4'                        ,
#     'abs(l2_dxy) > 0.01'                       ,
            'l2_reliso_rho_03 < 10'                    ,

            'hnl_q_12 == 0'                            ,
#     'abs(hnl_dphi_hnvis0) > 1.'                ,
            'nbj == 0'                                 ,
#     '!(hnl_w_vis_m > 50. & hnl_w_vis_m < 80.)' , # HACK! FIXME!
            '(hnl_w_vis_m > 50. & hnl_w_vis_m < 80.)' ,
            'hnl_dr_12 < 1.'                           ,
#     'hnl_q_01 == 0'                            ,
            'hnl_m_12 < 12'                            ,
            'sv_cos > 0.'                              ,

            # vetoes 12 (always OS anyways)
            'abs(hnl_m_12-3.0969) > 0.08'              , # jpsi veto
            'abs(hnl_m_12-3.6861) > 0.08'              , # psi (2S) veto
            'abs(hnl_m_12-0.7827) > 0.08'              , # omega veto
            'abs(hnl_m_12-1.0190) > 0.08'              , # phi veto

            # vetoes 01 (only is OS)
            '!(hnl_q_01==0 & abs(hnl_m_01-91.1876) < 10)'  , # Z veto
            '!(hnl_q_01==0 & abs(hnl_m_01- 9.4603) < 0.08)', # Upsilon veto
            '!(hnl_q_01==0 & abs(hnl_m_01-10.0233) < 0.08)', # Upsilon (2S) veto
            '!(hnl_q_01==0 & abs(hnl_m_01-10.3552) < 0.08)', # Upsilon (3S) veto
            '!(hnl_q_01==0 & abs(hnl_m_01-3.0969) < 0.08)', # jpsi veto
            '!(hnl_q_01==0 & abs(hnl_m_01-3.6861) < 0.08)', # psi (2S) veto
            '!(hnl_q_01==0 & abs(hnl_m_01-0.7827) < 0.08)', # omega veto
            '!(hnl_q_01==0 & abs(hnl_m_01-1.0190) < 0.08)', # phi veto

            # vetoes 02 (only is OS)
            '!(hnl_q_02==0 & abs(hnl_m_02-91.1876) < 10)'  , # Z veto
            '!(hnl_q_02==0 & abs(hnl_m_02- 9.4603) < 0.08)', # Upsilon veto
            '!(hnl_q_02==0 & abs(hnl_m_02-10.0233) < 0.08)', # Upsilon (2S) veto
            '!(hnl_q_02==0 & abs(hnl_m_02-10.3552) < 0.08)', # Upsilon (3S) veto
            '!(hnl_q_02==0 & abs(hnl_m_02-3.0969) < 0.08)', # jpsi veto
            '!(hnl_q_02==0 & abs(hnl_m_02-3.6861) < 0.08)', # psi (2S) veto
            '!(hnl_q_02==0 & abs(hnl_m_02-0.7827) < 0.08)', # omega veto
            '!(hnl_q_02==0 & abs(hnl_m_02-1.0190) < 0.08)', # phi veto
            
#     'hlt_IsoMu24'                              ,
#     'min(abs(hnl_dphi_01), abs(hnl_dphi_02))>1.', # dphi a la facon belgique
            'abs(hnl_dphi_01)>1',
            'abs(hnl_dphi_02)>1.', # dphi a la facon belgique
            
#     'sv_prob>0.005',
#     'hnl_2d_disp>=0.',

#     'abs(l1_dxy) > 0.005',
#     'abs(l2_dxy) > 0.005',

            'abs(l1_dxy) > 0.01',
            'abs(l2_dxy) > 0.01',
        ])

        self.selections['tight'] = ' & '.join([
             'l1_reliso_rho_03 < 0.2',
             'l2_reliso_rho_03 < 0.2',
        ])

        self.selections['is_prompt_lepton'] = ' & '.join([
            '(l1_gen_match_isPrompt==1 | l1_gen_match_pdgid==22)',
            '(l2_gen_match_isPrompt==1 | l2_gen_match_pdgid==22)',
#     'l1_gen_match_isPrompt==1',
#     'l2_gen_match_isPrompt==1',
        ])

        if self.channel == 'mmm':
            self.selections['zmm'] = ' & '.join([
                'l0_pt > 40'                               ,
                'abs(l0_eta) < 2.4'                        ,
                'abs(l0_dxy) < 0.05'                       ,
                'abs(l0_dz) < 0.2'                         ,
                'l0_reliso_rho_03 < 0.2'                   ,
                'l0_id_t == 1'                             ,

                'l1_pt > 35'                               ,
                'abs(l1_eta) < 2.4'                        ,
                'abs(l1_dxy) < 0.05'                       ,
                'abs(l1_dz) < 0.2'                         ,
                'l1_reliso_rho_03 < 0.2'                   ,
                'l1_id_t == 1'                             ,

                'hnl_q_01==0'                              ,
            ])

# convert to pandas readable queries
        self.selections_df = OrderedDict()
        for k, v in self.selections.items():
            vv = v.replace('&', 'and').replace('|', 'or').replace('!', 'not') 
            self.selections_df[k] = vv

