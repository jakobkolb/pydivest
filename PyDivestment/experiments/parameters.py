class ExperimentDefaults:
    """contains default values for experiment default parameters to keep them consistent"""

    input_params = {'b_c': 1., 'phi': .5, 'tau': 1.,
                    'eps': 0.05, 'b_d': 3, 'e': 1.,
                    'b_r0': 0.1,
                    'possible_cue_orders': [[0], [1]],
                    'xi': .05, 'beta': 0.06,
                    'L': 100., 'C': 1., 'G_0': 500000.,
                    'campaign': False, 'learning': True,
                    'interaction': 1, 'test': False,
                    'R_depletion': True}

    def calculate_timing(self, t_g, ):
        """calculate parameters of the economy to reproduce given timing"""
        pass
