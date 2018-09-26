

from pydivest.macro_model.integrate_equations_rep import Integrate_Equations as Rep
from pydivest.macro_model.integrate_equations_aggregate import IntegrateEquationsAggregate as Agg
from pydivest.micro_model.divestmentcore import DivestmentCore as Micro
from .run_model import run

models = {'Representative_agent': Rep,
          'Aggregate_approximation': Agg,
          'Micro_model': Micro}


for (name, model) in models.items():
    print(name, model)
    assert run(name, model) == 1
