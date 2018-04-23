

from pydivest.macro_model.integrate_equations_rep import Integrate_Equations as Rep
from pydivest.macro_model.integrate_equations_aggregate import Integrate_Equations as Agg
from pydivest.macro_model.integrate_equations_mean import Integrate_Equations as Mean
from .run_model import run

models = {'Representative_agent': Rep,
          'Aggregate_approximation': Agg,
          'Mean_approximation': Mean}


for (name, model) in models.items():
    print(name, model)
    assert run(name, model) == 1
