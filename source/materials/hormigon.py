from .LinearElastic import LinearElastic

E = 21e9       # Modulo de Young [Pa]
nu = 0.2       # Coeficiente de Poisson [-]
rho = 2400     # Densidad [kg/m3]

hormigon = LinearElastic(E, nu,plane_state="strain")  #Seleccionar el estado plano deseado "stress" o "strain"