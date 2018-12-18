class ExperimentDefaults:
    """contains default values for experiment default parameters to keep them consistent"""

    input_params = {'b_c': 1., 'b_d': 4., 'e': 1., 'b_r0': 0.1,
                    'kappa_c': .5, 'kappa_d': .5, 'pi': .5,
                    'xi': .1, 'd_k': 0.06, 'd_c': 0.1,
                    'phi': .5, 'tau': 1., 'eps': 0.03,
                    'possible_cue_orders': [[0], [1]],
                    'L': 100., 'C': 1., 'G_0': 1000000.,
                    'campaign': False, 'learning': True,
                    'interaction': 1, 'test': False,
                    'R_depletion': True}

    def calculate_timing(self, t_g, ):
        """calculate parameters of the economy to reproduce given timing"""
        pass
