import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

t_ref         = 298         #K          Reference temperature
delta_S0      = -162.86     #J/mol.K    Standard Entropy change
n             = 2           #           Number of electrons
F             = 96485       #C/mol      Faraday constant
R             = 8.315       #J/mol.K    Universal Gas Constant
delta_H0      = -286.02     #Kj/mol     Standard Enthalpy change
wt_percent    = 30          #%          Weight percent of KOH solution
gamma_a       = 1.25        #           Electrode Roughness Factor Anode
gamma_c       = 1.05        #           Electrode Roughness Factor Cathode
deltaG_a      = 41500       #J/mol.K    Free Energy of Activation Anode
deltaG_c      = 23450       #J/mol.K    Free Energy of Activation Cathode
i_o_ref_a     = 1.34535e-5  #A/cm2      Reference Exchange Current Density Anode
i_o_ref_c     = 1.8456e-3   #A/cm2      Reference Exchange Current Density Cathode
A_a           = 10000 #DESIGN VARIABLE      #cm2        Surface Area of Anode
A_c           = A_a         #cm2        Surface Area of Cathode
A_m           = 0.9*A_a     #cm2        Surface Area of Membrane
t_a           = 0.2         #cm         Thickness of Anode Electrode
t_c           = 0.2         #cm         Thickness of Cathode Electrode
t_m           = 0.05        #cm         Thickness of Membrane
rho_a         = 6.4e-6      #ohmcm      Resistivity of the 100%-Dense NI Electrode 
rho_c         = 6.4e-6      #ohmcm      Resistivity of the 100%-Dense NI Electrode 
k_a           = 0.00586     #/Celcius   Temperature Coefficient of Resistivity
k_c           = 0.00586     #/Celcius   Temperature Coefficient of Resistivity
d_am          = 0.125       #cm         Distance Between Anode and Membrane
d_cm          = 0.125       #cm         Distance Between Cathode and Membrane
eps_a         = 0.3         #           Porosity of Anode
eps_c         = 0.3         #           Porosity of Cathode
eps_m         = 0.42        #           Porosity of Membrane
tau_m         = 2.18        #           Tortuosity of Membrane
w_m           = 0.85        #           Separator Wettability Factor
j_lim         = 30          #A/cm2      Limiting current Density at 100 % bubble coverage 
N             = 1600  #DESIGN VARIABLE

U_tn = 1.48
erev = 1.229

T             = 273+75         #K
P             = 7           #bar
i             = 0.3  #DESIGN VARIABLE    #A/cm2


# Function to calculate the bubble coverage
def calculate_bubble_coverage(i,T,P,):
    #Vapor pressure of pure H2O
    term1 = np.log (T)
    #print("term1: ",term1)
    term2 = 37.043 - (6275.7 / T) - 3.4159*term1
    #print("term2: ",term2)
    p0_H2O = np.exp (term2)
    #p0_H2O = (T**-3.4159)*np.exp(37.043 - (6275.7/T))
    #print("p0_H2O: ",p0_H2O)
    term3 = np.exp (wt_percent / 115.96277)
    #print("term3: ",term3)
    m = (wt_percent * (183.1221 - (0.56845 * T) + (984.5679 * term3))) / 5610.5
    #print("m: ",m)
    #Vapor pressure of H2O
    term4 = 0.016214 - (0.13802 * m) + (0.19330 * m**0.5) + (1.0239 * np.log (p0_H2O))
    #print("term4: ",term4)
    pH2O = np.exp (term4)
    #print("pH2O: ",pH2O)
    #pH2O = (T**-3.498)*np.exp(37.93 - (6426.32/T))*np.exp(0.016214 - (0.13802 * m) + (0.19330 * m**0.5))

    #theta is the fractional bubble coverage. j_lim is the limiting current density at 100% bubble coverage.
    theta = (-97.25 + 182 * (T / t_ref) - 84 * (T / t_ref)**2) * ((i / j_lim)**0.3) * (P / (P - pH2O))

    #print (theta)
    return theta
    

# Function to calculate reversible voltage
def calculate_reversible_voltage(T,P):
    #Vapor pressure of pure H2O
    term1 = np.log (T)
    term2 = 37.043 - (6275.7 / T) - 3.4159*term1
    p0_H2O = np.exp (term2)
    #p0_H2O = (T**-3.4159)*np.exp(37.043 - (6275.7/T))
    
    term3 = np.exp (wt_percent / 115.96277)
    m = (wt_percent * (183.1221 - (0.56845 * T) + (984.5679 * term3))) / 5610.5
    
    #Vapor pressure of H2O
    term4 = 0.016214 - (0.13802 * m) + (0.19330 * m**0.5) + (1.0239 * np.log (p0_H2O))
    pH2O = np.exp (term4)
    #pH2O = (T**-3.498)*np.exp(37.93 - (6426.32/T))*np.exp(0.016214 - (0.13802 * m) + (0.19330 * m**0.5))
    
    erev = 1.229
    
    term5 = ((P - pH2O)**1.5) * p0_H2O/ pH2O
    V_rev = erev + ((T - t_ref)*(-0.9e-3)) + (((R * T) / (2 * F)) * np.log(term5))
    
    return V_rev


# Function to calculate activation overpotential
def calculate_activation_overpotential(i,T,P):
    


    #alpha_a,c is the charge transfer co-efficient of anode and cathode
    alpha_a = 0.0675 + 0.00095 * T
    alpha_c = 0.1175 + 0.00095 * T
    

    #b_a,c is the "Tafel Slope". The slope of activation overpotential in linear region of polarization curve is determined by tafel slope
    b_a = (R * T) / (n * F * alpha_a)
    b_c = (R * T) / (n * F * alpha_c)
    
    #i_o_a,c is the exchange current density. gamma_a,c is the reference exchange current density at Reference Temperature.
    i_o_a = gamma_a * i_o_ref_a * np.exp((-deltaG_a / R) * ((1 / T) - (1 / t_ref)))
    i_o_c = gamma_c * i_o_ref_c * np.exp((-deltaG_c / R) * ((1 / T) - (1 / t_ref)))


    theta=calculate_bubble_coverage(i,T,P)    
    
    
    #Activation Overpotential
    V_act_anode = b_a * np.log (i / i_o_a) + b_a * np.log(1 / (1 - theta))
    V_act_cathode = b_c * np.log (i / i_o_c) + b_c * np.log(1 / (1 - theta))
    
    V_act = V_act_anode + V_act_cathode
    return V_act


# Function to calculate ohmic overpotential
def calculate_ohmic_overpotential(i,T,P): 
    
    theta=calculate_bubble_coverage(i,T,P)

    term3 = np.exp (wt_percent / 115.96277)
    m = (wt_percent * (183.1221 - (0.56845 * T) + (984.5679 * term3))) / 5610.5
    
    #Effective Resistivity of anode and cathode. eps_a,c is the resistivity of 100% dense cathode,anode an Reference Temperature.
    rho_eff_a = rho_a / ((1 - eps_a)**1.5)
    rho_eff_c = rho_c / ((1 - eps_c)**1.5)
    
    #Electrode Resistance
    R_ele = ((rho_eff_a * t_a / A_a) * (1 + k_a * (T - t_ref))) + ((rho_eff_c * t_c / A_c) * (1 + k_c * (T - t_ref)))
    
    #sigma_KOH_free is the tortuosity of memebrane
    sigma_KOH_free = (-2.04 * m) - (0.0028 * m**2) + (0.005332 * m * T) + (207.2 * m / T) + (0.001043 * m**3) - (0.0000003 * m**2 * T**2)
    
    #Bubble Free electrolyte resistance
    R_ely_free = (1 / sigma_KOH_free) * ((d_am / A_a) + (d_cm / A_c))
    
    #Electrolyte Resistance with bubbles
    R_ely_bubble = R_ely_free * ((1 / (1 - ((2/3)*theta))**1.5)-1)
    
    #Electrolyte Resistance = Sum of electrolytic resistance with and without bubble formation
    R_ely = R_ely_free + R_ely_bubble
    
    #Memebrane Resistance
    R_mem = (1 / sigma_KOH_free) * ((tau_m**2 * t_m) / (w_m * eps_m * A_m))
    
    I = i * (A_a)
    
    V_ohmic = I * (R_ele + R_ely + R_mem)
    return V_ohmic


# Function to calculate cell voltage
def calculate_cell_voltage(i,T,P):
    V_rev = calculate_reversible_voltage(T,P)
    V_act = calculate_activation_overpotential(i,T,P)
    V_ohmic = calculate_ohmic_overpotential(i,T,P)
    
    V_cell = (V_rev+ V_act+ V_ohmic)
    
    return V_cell


# Function to calculate hydrogen evolution rate
def calculate_hydrogen_evolution_rate(i):
    # Define hydrogen evolution rate calculation here
    return i * A_c * N / (2 * F)  # Rate of hydrogen evolution (mol/sec)


# Function to calculate oxygen evolution rate
def calculate_oxygen_evolution_rate(i):
    # Define oxygen evolution rate calculation here
    return i * A_a * N / (4 * F)  # Rate of oxygen evolution (mol/sec)


# Current density range
current_density_range = np.linspace(0, 0.5, 100)[1:]  # A/cm^2


# Temperature range
temperature_range = [273+30, 273+40, 273 + 50, 273 + 60, 273 + 70, 273 + 80, 273 + 90, 273 + 100]  # K


# Temperature range
temperature_range = [273+30, 273+40, 273 + 50, 273 + 60, 273 + 70, 273 + 80, 273 + 90, 273 + 100]  # K


# Accumulate bubble coverage for different temperatures
bubble_coverage_all_temps = []
for T in temperature_range:
    bubble_coverage = [calculate_bubble_coverage(i,T,P) for i in current_density_range]
    bubble_coverage_all_temps.append(bubble_coverage)


# Pressure range
pressure_range = [2, 4, 6, 8, 10]  # bar


# Accumulate cell voltage for different temperatures
cell_voltage_all_pressures = []
for P in pressure_range:
    cell_voltage = [calculate_cell_voltage(i,T,P) for i in current_density_range]
    cell_voltage_all_pressures.append(cell_voltage)



# Calculate cell voltage for different current densities
V_reversible = calculate_reversible_voltage(T,P)
activation_overpotential = [calculate_activation_overpotential(i,T,P) for i in current_density_range]
ohmic_overpotential = [calculate_ohmic_overpotential(i,T,P) for i in current_density_range]
cell_voltage = [calculate_cell_voltage(i,T,P) for i in current_density_range]
hydrogen_prod_rate = [calculate_hydrogen_evolution_rate(i) for i in current_density_range]

Total_hydrogen_production = [calculate_hydrogen_evolution_rate(i) for i in current_density_range * 86400 * 2 / 1000]

Faraday_efficiency =  [calculate_hydrogen_evolution_rate(i) * 2 * F / (N * i * A_c) for i in current_density_range] 

Energy_efficiency =  [U_tn / calculate_cell_voltage(i,T,P) for i in current_density_range]


current_density_range = np.linspace(0, 0.5, 100)[1:] 
temperature_range = [273+30, 273+40, 273 + 50, 273 + 60, 273 + 70, 273 + 80, 273 + 90, 273 + 100]  # K
pressure_range = [2, 4, 6, 8, 10]


data=[]
for T in temperature_range:
    for P in pressure_range:
        for i in current_density_range:
            data.append([T,P,i])



df=pd.DataFrame(data,columns=["Temperature","Pressure","Current Density"])

bubble_coverage_list=[]
reversible_voltage_list=[]
activation_overpotential_list=[]
ohmic_overpotential_list=[]
cell_voltage_list=[]
hydrogen_evolution_rate_list=[]
oxygen_evolution_rate_list=[]


for k in range (len(df)):
    current_density = df["Current Density"][k]
    temperature = df["Temperature"][k]
    pressure = df["Pressure"][k]
    
    bubble_coverage = calculate_bubble_coverage(current_density,temperature,pressure)
    bubble_coverage_list.append(bubble_coverage)
    
    reversible_voltage = calculate_reversible_voltage(temperature,pressure)
    reversible_voltage_list.append(reversible_voltage)
    
    activation_overpotential = calculate_activation_overpotential(current_density,temperature,pressure)
    activation_overpotential_list.append(activation_overpotential)
    
    ohmic_overpotential = calculate_ohmic_overpotential(current_density,temperature,pressure)
    ohmic_overpotential_list.append(ohmic_overpotential)
    
    cell_voltage = calculate_cell_voltage(current_density,temperature,pressure)
    cell_voltage_list.append(cell_voltage)
    
    hydrogen_evolution_rate = calculate_hydrogen_evolution_rate(current_density)
    hydrogen_evolution_rate_list.append(hydrogen_evolution_rate)

    oxygen_evolution_rate = calculate_oxygen_evolution_rate(current_density)
    oxygen_evolution_rate_list.append(oxygen_evolution_rate)



df["Bubble Coverage"]=bubble_coverage_list
df["Reversible Voltage"]=reversible_voltage_list
df["Activation Overpotential"]=activation_overpotential_list
df["Ohmic Potential"]=ohmic_overpotential_list
df["Cell Voltage"]=cell_voltage_list
df["Hydrogen Evolution Rate"]=hydrogen_evolution_rate_list
df["Oxygen Evolution Rate"]=oxygen_evolution_rate_list

df = df.sample(frac = 1).reset_index(drop=True)


#AutoEncoder Model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
import optuna


# Define custom Dataset
class BubbleCoverageDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y
    

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        weighted_output = x * attention_weights
        return weighted_output
    


# Define Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output is a single value for Bubble Coverage
        )
        self.attention = Attention(encoding_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        attended = self.attention(encoded)
        encoded_with_residual = encoded + attended
        decoded = self.decoder(encoded_with_residual)
        #decoded = self.decoder(encoded)
        return decoded
    

def train_autoencoder(df, epochs=100, batch_size=32, learning_rate=0.001, encoding_dim=32, weight_decay=0.0):
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Split dataset into train and test
    features = df[['Temperature', 'Pressure', 'Current Density', 'Bubble Coverage',
       'Reversible Voltage', 'Activation Overpotential', 'Ohmic Potential',
       'Cell Voltage']].values
    targets = df['Hydrogen Evolution Rate'].values
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

    # Prepare datasets and dataloaders
    train_dataset = BubbleCoverageDataset(X_train, y_train)
    test_dataset = BubbleCoverageDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, loss, and optimizer
    model = Autoencoder(input_dim=8, encoding_dim=encoding_dim).to(device)
    criterion = nn.MSELoss()
    #criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


    train_losses = []
    test_r2_scores = []
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        #print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

    return model, train_loader, test_loader, device, train_losses


# Testing function
from sklearn.metrics import mean_absolute_error, r2_score
def test_autoencoder(model, train_loader, test_loader, device):
    model.eval()
    train_targets, train_predictions = [], []
    test_targets, test_predictions = [], []

    # Evaluate on training data
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()
            train_predictions.extend(outputs.cpu().numpy())
            train_targets.extend(targets.cpu().numpy())

    # Evaluate on testing data
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()
            test_predictions.extend(outputs.cpu().numpy())
            test_targets.extend(targets.cpu().numpy())

    # Calculate metrics
    train_mae = mean_absolute_error(train_targets, train_predictions)
    train_r2 = r2_score(train_targets, train_predictions)
    test_mae = mean_absolute_error(test_targets, test_predictions)
    test_r2 = r2_score(test_targets, test_predictions)

    return {
        "train_mae": train_mae,
        "train_r2": train_r2,
        "test_mae": test_mae,
        "test_r2": test_r2
    }


# Optuna objective function

def objective(trial):
    # Hyperparameters to tune
    encoding_dim = trial.suggest_int("encoding_dim", 16, 64, step=16)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int("batch_size", 32, 128, step=16)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)

    # Split data into train/test
    features = df[['Temperature', 'Pressure', 'Current Density']].values
    targets = df['Bubble Coverage'].values
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

    # Prepare datasets and dataloaders
    train_dataset = BubbleCoverageDataset(X_train, y_train)
    test_dataset = BubbleCoverageDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(input_dim=3, encoding_dim=encoding_dim).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop
    for epoch in range(50):  # Fixed epochs for tuning
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate on test set
    model.eval()
    test_targets, test_predictions = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()
            test_predictions.extend(outputs.cpu().numpy())
            test_targets.extend(targets.cpu().numpy())

    # Metrics
    test_r2 = r2_score(test_targets, test_predictions)
    return -test_r2  # Negative R² because Optuna minimizes the objective


def plot_loss(train_losses,name):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, linestyle='-')
    plt.title('Training Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.xticks(range(1, 200 + 1, 10))
    plt.tight_layout() 
    plt.title(name)
    plt.savefig(f"{name}_AutoEncoderModel.jpg")
    plt.show()
    

# Optuna Study
# Run Optuna optimization
study = optuna.create_study(study_name="Optuna_Study", direction="minimize")
study.optimize(objective, n_trials=100)

# Print best hyperparameters
print("Best hyperparameters:", study.best_params)

# Train final model using the best hyperparameters
best_params = study.best_params


final_model, train_loader, test_loader, device, train_losses_optuna = train_autoencoder(
    df,
    epochs=200,  # Train longer with the best params
    batch_size=best_params["batch_size"],
    learning_rate=best_params["learning_rate"],
    encoding_dim=best_params["encoding_dim"],
    weight_decay=best_params["weight_decay"]
)



# Test the model
metrics = test_autoencoder(final_model, train_loader, test_loader, device)

print("Optuna Params")
print(f"Final Train MAE: {metrics['train_mae']:.4f}, Train R²: {metrics['train_r2']:.4f}")
print(f"Final Test MAE: {metrics['test_mae']:.4f}, Test R²: {metrics['test_r2']:.4f}")





#Personalized Study
final_model, train_loader, test_loader, device, train_losses = train_autoencoder(
    df,
    epochs=200,  # Train longer with the best params
    batch_size=32,
    learning_rate=0.00109128955824676,
    encoding_dim=48,
    weight_decay=1.8049789124923568e-05
)



# Test the model
metrics = test_autoencoder(final_model, train_loader, test_loader, device)

print("Personalized Params")
print(f"Final Train MAE: {metrics['train_mae']:.4f}, Train R²: {metrics['train_r2']:.4f}")
print(f"Final Test MAE: {metrics['test_mae']:.4f}, Test R²: {metrics['test_r2']:.4f}")


#Plotting graphs 
plot_loss(train_losses_optuna,"Optuna Params")
plot_loss(train_losses,"Personlized Params")
