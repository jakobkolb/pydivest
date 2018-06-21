class ExperimentDefaults:
    """contains default values for experiment default parameters to keep them consistent"""

    input_params = {'b_c': 1., 'phi': .4, 'tau': 1.,
                    'eps': 0.05, 'b_d': 1.25, 'e': 100.,
                    'b_r0': 0.1 ** 2 * 100.,
                    'possible_cue_orders': [[0], [1]],
                    'xi': 1. / 8., 'beta': 0.06,
                    'L': 100., 'C': 100., 'G_0': 800.,
                    'campaign': False, 'learning': True,
                    'interaction': 1, 'test': False}

    def calculate_timing(self, t_g, ):
        """calculate parameters of the economy to reproduce given timing"""
        pass
