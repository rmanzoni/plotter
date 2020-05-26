from collections import OrderedDict

class Selections(object):

    def __init__(self, channel):
        self.channel = channel
        self.base    = None

        self.selections = OrderedDict()

        if self.channel == 'mmm':
            self.selections['pt_iso'] = ' & '.join(['l0_pt > 25'      ,
                                                    'l2_pt > 5'       ,
                                                    'l1_pt > 5'       ,
                                                    'l0_id_m == 1'    ,
                                                    'l1_id_hnl_m == 1',
                                                    'l2_id_hnl_m == 1',])

        if self.channel == 'mem':
            self.selections['pt_iso'] = ' & '.join(['l0_pt > 25'            ,
                                                    'l2_pt > 5'             ,
                                                    'l1_pt > 5'             ,
                                                    'l0_id_m == 1'          ,
                                                    'l1_id_hnl_l_niso == 1' ,
                                                    'l2_id_hnl_m == 1'      ,])

        if self.channel == 'eem':
            self.selections['pt_iso'] = ' & '.join(['l0_pt > 32'            ,
                                                    'l2_pt > 5'             ,
                                                    'l1_pt > 5'             ,
                                                    'l0_id_mva_niso_90 == 1',
                                                    'l1_id_hnl_l_niso == 1' ,
                                                    'l2_id_hnl_m == 1'      ,])

        if self.channel == 'eee':
            self.selections['pt_iso'] = ' & '.join(['l0_pt > 32'            ,
                                                    'l2_pt > 5'             ,
                                                    'l1_pt > 5'             ,
                                                    'l0_id_mva_niso_90 == 1',
                                                    'l1_id_hnl_l_niso == 1' ,
                                                    'l2_id_hnl_l_niso == 1' ,])

        assert self.selections['pt_iso'], 'Error: No channel specific selection applied!'

        self.selections['pre_baseline'] = ' & '.join([
            'abs(l0_eta) < 2.4'     ,
            'abs(l0_dxy) < 0.05'    ,
            'abs(l0_dz) < 0.1'      ,
            'l0_reliso_rho_03 < 0.1',

            'abs(l1_eta) < 2.4'     ,
            'l1_reliso_rho_03 < 10' ,

            'abs(l2_eta) < 2.4'     ,
            'l2_reliso_rho_03 < 10' ,

            'hnl_q_12 == 0'         ,

            'hnl_dr_12 < 1.'        ,
            'hnl_dr_12 > 0.02'      ,

            'hnl_m_12 < 20'         ,
            
            'abs(hnl_dphi_01)>1.'   ,
            'abs(hnl_dphi_02)>1.'   , # dphi a la facon belgique
            
            'pass_met_filters==1'   ,
            ])

        self.selections['baseline'] = ' & '.join([
            self.selections['pre_baseline'],
            'nbj == 0'                     ,
            'hnl_2d_disp_sig>20'           ,
            'hnl_pt_12>15'                 ,
            'sv_cos>0.99'                  ,
            'sv_prob>0.001'                ,
            'abs(l1_dz)<10'                ,
            'abs(l2_dz)<10'                ,
            'abs(l1_dxy) > 0.01'           ,
            'abs(l2_dxy) > 0.01'           ,
        ])

        self.selections['sideband'] = '!(hnl_w_vis_m > 50. & hnl_w_vis_m < 80.)' # THIS IS IMPORTANT!

        self.selections['signal_region'] = '(hnl_w_vis_m > 50. & hnl_w_vis_m < 80.)' # THIS IS IMPORTANT!

#         self.selections['vetoes_12_OS'] = ' & '.join([
#             # vetoes 12 (always OS anyways)
#             'abs(hnl_m_12-3.0969) > 0.08'              , # jpsi veto
#             'abs(hnl_m_12-3.6861) > 0.08'              , # psi (2S) veto
#             'abs(hnl_m_12-0.7827) > 0.08'              , # omega veto
#             'abs(hnl_m_12-1.0190) > 0.08'              , # phi veto
#             ])

        # after discussing with Martina 9/1/2020
        self.selections['vetoes_12_OS'] = ' & '.join([
            # vetoes 12 (always OS anyways)
            '!(hnl_2d_disp<1.5 & abs(hnl_m_12-3.0969) < 0.08)', # jpsi veto
            '!(hnl_2d_disp<1.5 & abs(hnl_m_12-3.6861) < 0.08)', # psi (2S) veto
            '!(hnl_2d_disp<1.5 & abs(hnl_m_12-0.7827) < 0.08)', # omega veto
            '!(hnl_2d_disp<1.5 & abs(hnl_m_12-1.0190) < 0.08)', # phi veto
            ])

           
        self.selections['vetoes_01_OS'] = ' & '.join([
            # vetoes 01 (only is OS)
            '!(hnl_q_01==0 & abs(hnl_m_01-91.1876) < 10)'  , # Z veto
            '!(hnl_q_01==0 & abs(hnl_m_01- 9.4603) < 0.08)', # Upsilon veto
            '!(hnl_q_01==0 & abs(hnl_m_01-10.0233) < 0.08)', # Upsilon (2S) veto
            '!(hnl_q_01==0 & abs(hnl_m_01-10.3552) < 0.08)', # Upsilon (3S) veto
            '!(hnl_q_01==0 & abs(hnl_m_01-3.0969)  < 0.08)', # jpsi veto
            '!(hnl_q_01==0 & abs(hnl_m_01-3.6861)  < 0.08)', # psi (2S) veto
            '!(hnl_q_01==0 & abs(hnl_m_01-0.7827)  < 0.08)', # omega veto
            '!(hnl_q_01==0 & abs(hnl_m_01-1.0190)  < 0.08)', # phi veto
            ])

        self.selections['vetoes_02_OS'] = ' & '.join([
            # vetoes 02 (only is OS)
            '!(hnl_q_02==0 & abs(hnl_m_02-91.1876) < 10)'  , # Z veto
            '!(hnl_q_02==0 & abs(hnl_m_02- 9.4603) < 0.08)', # Upsilon veto
            '!(hnl_q_02==0 & abs(hnl_m_02-10.0233) < 0.08)', # Upsilon (2S) veto
            '!(hnl_q_02==0 & abs(hnl_m_02-10.3552) < 0.08)', # Upsilon (3S) veto
            '!(hnl_q_02==0 & abs(hnl_m_02-3.0969)  < 0.08)', # jpsi veto
            '!(hnl_q_02==0 & abs(hnl_m_02-3.6861)  < 0.08)', # psi (2S) veto
            '!(hnl_q_02==0 & abs(hnl_m_02-0.7827)  < 0.08)', # omega veto
            '!(hnl_q_02==0 & abs(hnl_m_02-1.0190)  < 0.08)', # phi veto
            ])

        self.selections['tight'] = ' & '.join([
             'l1_reliso_rho_03 < 0.2',
             'l2_reliso_rho_03 < 0.2',
            ])

        # RM  is this wrong? this allows for one of the two displaced leptons to be 
        # neither prompt nor conversion
#         self.selections['is_prompt_lepton'] = '(%s)' %(' | '.join([
#             'l1_gen_match_isPrompt==1',
#             'l1_gen_match_pdgid==22',
#             'l2_gen_match_isPrompt==1',
#             'l2_gen_match_pdgid==22',
#             ]))

        self.selections['is_prompt_lepton'] = ' & '.join([
            '(l1_gen_match_isPrompt==1 | l1_gen_match_pdgid==22)',
            '(l2_gen_match_isPrompt==1 | l2_gen_match_pdgid==22)',
            ])

        self.selections['zmm'] = ' & '.join([
            'l0_pt > 40'            ,
            'abs(l0_eta) < 2.4'     ,
            'abs(l0_dxy) < 0.05'    ,
            'abs(l0_dz) < 0.2'      ,
            'l0_reliso_rho_03 < 0.2',
            'l0_id_t == 1'          ,

            'l1_pt > 35'            ,
            'abs(l1_eta) < 2.4'     ,
            'abs(l1_dxy) < 0.05'    ,
            'abs(l1_dz) < 0.2'      ,
            'l1_reliso_rho_03 < 0.2',
            'l1_id_t == 1'          ,

            'hnl_q_01==0'           ,
            
            'abs(hnl_dphi_01)>1.'   ,

            'pass_met_filters==1'   ,
            ])

        self.selections['zee'] = ' & '.join([
            'l0_pt > 40'            ,
            'abs(l0_eta) < 2.4'     ,
            'abs(l0_dxy) < 0.05'    ,
            'abs(l0_dz) < 0.2'      ,
            'l0_reliso_rho_03 < 0.2',
            'l0_id_mva_niso_90 == 1'          ,

            'l1_pt > 35'            ,
            'abs(l1_eta) < 2.4'     ,
            'abs(l1_dxy) < 0.05'    ,
            'abs(l1_dz) < 0.2'      ,
            'l1_reliso_rho_03 < 0.2',
            'l1_id_mva_niso_90 == 1',

            'hnl_q_01==0'           ,
            
            'abs(hnl_dphi_01)>1.'   ,

            'pass_met_filters==1'   ,
            ])

        # convert to pandas readable queries
        self.selections_pd = OrderedDict()
        for k, v in self.selections.items():
            vv = v.replace('&', 'and').replace('|', 'or').replace('!', 'not') 
            self.selections_pd[k] = vv
