import os
import uproot
import awkward as ak
import numpy as np
import random
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Function
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, roc_curve, auc
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import label_binarize

seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed) 

##for recognizing overflow, underflow
def print_bin_stats(values, edges, tag):
    """
    values: 1D numpy array of numbers (your feature values)
    edges : 1D numpy array of bin edges (length = n_bins + 1)
    tag   : label for the printout, e.g. 'train_sig'
    """
    # per-bin counts
    counts, _ = np.histogram(values, bins=edges)

    # under/overflow
    under = (values <  edges[0]).sum()
    over  = (values >= edges[-1]).sum()
    total = values.size

    print(f"{tag}: total={total}, underflow={under}, overflow={over}")
    print(f"  per-bin counts: {counts.tolist()}")
    return counts, under, over

#########SETS THE NUMBER OF ADVERSARIAL CLASSES
nbins=2

##reads command-line flag --epochs or -e to set how many passes through the data to make during training
parser = argparse.ArgumentParser(description="describe epoch numbers plot")
parser.add_argument("--epochs", "-e", type=int, default=5, help="Number of training epochs")
parser.add_argument("--hidden_dims", nargs="*", help="Hidden layers dimensions", type=int, default=[])
parser.add_argument("--dropout", help="dropout", type=float, default=0.0)
parser.add_argument("--activation", default='relu')
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()
EPOCH=args.epochs

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

##Gradient Reversal Layer
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    ##in backward pass multiply all gradients by -lambda, classifier pushes features that fool the adversary
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_=1.0):
    return GradReverse.apply(x, lambda_)


####MAIN NEURAL NETWORK FOR CLASSIFICATION
##input layer: take input_dim features per event
##hidden layer: 64 neurons with a ReLu activation - give non linearity
##output layer: a logit per event -> prob of signal vs bkg

#TO DO: get classifier object to extract hidden layer info, keep its ability to more than one hidden layer, i want certain dim and extract their nodes
#TO DO: want the number associated than the matrix, output after eval number itself extract
#TO DO: not care about weights with the matrices, actual calculation the node does, weight + inputs
class Classifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims=[], dropout=0, activation='relu'):
        super().__init__()

        layers = []

        prev_dim = input_dim
        for hidden_dim_idx, hidden_dim in enumerate(hidden_dims):
            if dropout and hidden_dim_idx > 0:
                layers.append(torch.nn.Dropout(p=dropout))
            layers += [
                torch.nn.Linear(prev_dim, hidden_dim),
                {'relu': torch.nn.ReLU(), 'leaky_relu': torch.nn.LeakyReLU(), 'tanh': torch.nn.Tanh(), 'sigmoid': torch.nn.Sigmoid()}[activation]
            ]
            prev_dim = hidden_dim

        self.hidden_net = torch.nn.Sequential(*layers)
        self.output_layer = torch.nn.Linear(prev_dim, 1)

    def forward(self, x):
        coding = self.hidden_net(x)
        if torch.isnan(coding).any() or torch.isinf(coding).any():
            print('[DIAG] NaNs or Infs in coding')
        out = self.output_layer(coding).squeeze(-1)             # classifier output
        return out, coding                               # return both

######OTHER NEURAL NETWORK FOR ADVERSARIAL TASK
##takes as input the classifier's internal representation - raw logit after applying gradient reversal
##passes it through a small network (32-unit hiddent layer) and outputs nbins scores
##nbins scores correspond to your 5 bins (the "adversarial" task of predicting which psum-range bin an event falls into)

## longer explaination: For each event, classifier produces a single number called a logit - unsquashed "confidence" score for signal vs. background.
## We take that logit, wrap it in a tiny vector, and feed it to adversary network. Before adversary's layers see data,
## pass it thru gradient reversal operation. In the forward direction, gradient reversal is the identity function - return logit
## unchanged. The adversary processes that logit through 32-neuron hidden layer + ReLu and a 5-neuron output layer.
## The five numbers are raw score (one per bin) that the adversary will use to predict which of the five psum ranges the original event came from.

class Adversary(torch.nn.Module):
    def __init__(self, input_dim):
         #ADV-ED COMMENTED OUT
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, nbins)  # regression on psum/pz
        )

    def forward(self, x, lambda_ = 1): # #ADV-ED COMMENTED OUT CHANGED FROM JUST LAMBDA_ TO LAMBDA_ = 1
        x = grad_reverse(x, lambda_)
        return self.net(x)

## specifies which physics variables to pull from two ROOT files (one 0 and another 1)
## prepares empty lists to accumulate your features, signal/background labels, and adversary-bin labels

# Only load specific branches
#############THIS CONTAINS ALL THE BRANCHES OF THE ROOT FILE WE AIM TO USE; ADD MORE NAMES AS YOU SEE FIT (OR REMOVE)#############
branches=[]
branches.extend(["psum","vertex.invM_","vertex.pos_"])
#,"pos.track_.track_time_",, "ele.track_.track_time_"
branches.extend(["ele.track_.n_hits_", "ele.track_.d0_", "ele.track_.phi0_", "ele.track_.z0_", "ele.track_.tan_lambda_", "ele.track_.px_", "ele.track_.py_", "ele.track_.pz_", "ele.track_.chi2_","ele.track_.x_at_ecal_","ele.track_.y_at_ecal_","ele.track_.z_at_ecal_"])
branches.extend(["pos.track_.n_hits_", "pos.track_.d0_", "pos.track_.phi0_", "pos.track_.z0_", "pos.track_.tan_lambda_", "pos.track_.px_", "pos.track_.py_", "pos.track_.pz_", "pos.track_.chi2_","pos.track_.x_at_ecal_","pos.track_.y_at_ecal_","pos.track_.z_at_ecal_" ])
branches.extend(["vertex.chi2_","vertex.invMerr_"])
branches.extend(["vtx_proj_sig","vtx_proj_x_sig","vtx_proj_y_sig"])
#branches.extend(["ele.track_.track_residuals_[14]","pos.track_.track_residuals_[14]"])
#branches.extend(["ele.track_.lambda_kinks_[14]","pos.track_.lambda_kinks_[14]"])
#branches.extend(["ele.track_.phi_kinks_[14]","pos.track_.phi_kinks_[14]"])

###############THIS ESTABLISHES WHAT BRANCH WE ARE MAKING ADVERSARIAL########
cut_branch = "vertex.invM_"
cut_threshold = 3.0
files_and_labels = [
    ("merged_tritrig_pulser_recon_tomstyle.root",0), #background
    ("merged_simp_pulser_allMasses_recon_tomstyle.root",1) #physics signal allMasses=mass invariant
]

all_data = []
all_labels = []
cut_labels = []
flattened_names = []  # to be populated once from first file


## opens each file's preselection TTree, reads only the desired branches into memory, applies a mask,
## flattens each branch into a single 2D feature array, labels each event as signal (1) or background (0)
## divides the chosen variable (psum) range into 5 equal bins and assigns each event a bin index from 0 to 4.
## accumulates all data, labels, and bin-labels

###for smaller mass
### when fills up to 10,000 events move on to next file
#################THIS LOADS IN ALL THE EVENTS FROM THE SIGNAL AND BACKGROUND FILE WITH A MASK IF YOU WANT DISPLACED VERTICES#############
for filename, label in files_and_labels:
    #TO SKIP 10K EVENTS event_count = 0
    with uproot.open(f"{filename}:preselection") as tree:
        #print(tree.keys())
        arrays = tree.arrays(branches)
        invM = ak.to_numpy(arrays["vertex.invM_"])
        ARR1 = ak.to_numpy(arrays["vertex.pos_"])
        #for field in ARR1.dtype.names:
        #    print(field)
        z = ARR1["fZ"]
        psum_values = ak.to_numpy(arrays[cut_branch])
        #############THIS LINE DOES A CUT ON DISPLACE VERTICES LENGTH (Z) AS WELL AS INVARIANT MASS (THOUGH THAT CUT IS SO LOOSE AS TO BE VOID)############
        cut_mask = (invM>-100) & (invM<.18)# & (z>10.0) #(invM > .001*mass-.005) & (invM < .001*mass+.005)
        if np.sum(cut_mask) == 0:
            continue  # skip empty batch

        data_parts = []
        feature_names = []
        #############DIFFERENT BRANCHES HAD WIDELY VARYING STRUCTURE, THIS LOOP IS MEANT TO STORE EACH AS A 1D VARIABLE##########
        for b in branches:
            arr = ak.to_numpy(arrays[b])[cut_mask]
            
            # Handle structured dtypes (e.g., vertex.pos_)
            if arr.dtype.fields is not None:
                for field in arr.dtype.names:
                    subarr = arr[field]
                    if not np.issubdtype(subarr.dtype, np.number):
                        print(f"Skipping {b}.{field}: non-numeric dtype {subarr.dtype}")
                        continue
                    data_parts.append(subarr.reshape(-1, 1))
                    feature_names.append(f"{b}.{field}")
            elif np.issubdtype(arr.dtype, np.number):
                if arr.ndim == 1:
                    data_parts.append(arr.reshape(-1, 1))
                    feature_names.append(b)
                elif arr.ndim == 2:
                    for i in range(arr.shape[1]):
                        data_parts.append(arr[:, i].reshape(-1, 1))
                        feature_names.append(f"{b}[{i}]")
                else:
                    print(f"Skipping {b}: unsupported shape {arr.shape}")
            else:
                print(f"Skipping {b}: non-numeric dtype {arr.dtype}")
        #for i, part in enumerate(data_parts):
            #print(f"Array {i} shape: {part.shape}")

        data = np.hstack(data_parts)
        labels = np.full(data.shape[0], label)

        ############THIS STORES THE DATA INTO all_data ALONGSIDE labels TO TELL YOU SIGNAL OR BACKGROUND AND cut_labels FOR WHAT ADVERSARY REGION YOU'RE IN
        # Example: 5 bins (adjust as needed)
        n_bins = nbins
        print(psum_values[cut_mask].min())
        print(psum_values[cut_mask].max())
        bins = np.linspace(psum_values[cut_mask].min(), psum_values[cut_mask].max(), n_bins + 1)
        cut_label = np.digitize(psum_values[cut_mask], bins=bins) - 1
        cut_label = np.clip(cut_label, 0, n_bins - 1)

        all_data.append(data)
        all_labels.append(labels)
        cut_labels.append(cut_label)

        if not flattened_names:
            flattened_names = feature_names

###########THIS IS THE MEAT OF THE CODE, THE PORTION DOING TRAINING. FIRST WE MUST STAGE THE DATA A BIT#############

# Combine everything
## concatenates all events into single feature matrix X, signal label vector Y, and bin label vector Z.
## splits into training (70%) and testing (30%), making sure the fraction of signal vs. background stays the same in both sets.
## wraps them in PyTorch DataLoaders so you can iterate in minibatches of 128 events.
X = np.concatenate(all_data)
Y = np.concatenate(all_labels)
Z = np.concatenate(cut_labels)

print('X:', X.shape, 'Y:', Y.shape, 'Z:', Z.shape   )

# Create a composite label for stratification; this line will keep the relative ratios of signal and background as well as control and not control constant.

composite_labels = np.array([f"{y}_{z}" for y, z in zip(Y, Z)])
####################THIS ESTABLISHED THE TRAIN AND TESTING SET, BY SETTING ASIDE RANDOM EVENTS AT A RATE OF 1/3 FOR TRAINING, IT KEEPS Y (THE AMOUNT OF SIGNAL) PROPORTIONAL IN EACH SET###############
X_train, X_tmp, Y_train, Y_tmp, Z_train, Z_tmp = train_test_split(
    X, Y, Z, test_size=0.3, random_state=42, stratify=Y
)

X_val, X_test, Y_val, Y_test, Z_val, Z_test =  train_test_split(
    X_tmp, Y_tmp, Z_tmp, test_size=0.5, random_state=42, stratify=Y_tmp
)

print('X_train:', X_train.shape, 'X_val:', X_val.shape, 'X_test:', X_test.shape)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(Y_train, dtype=torch.long),
                              torch.tensor(Z_train, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)

val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                              torch.tensor(Y_val, dtype=torch.long),
                              torch.tensor(Z_val, dtype=torch.long))
val_loader = DataLoader(val_dataset, batch_size=128, drop_last=True)

test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(Y_test, dtype=torch.long),
                             torch.tensor(Z_test, dtype=torch.long))
test_loader = DataLoader(test_dataset, batch_size=128, drop_last=True)

##########WE NOW ESTABLISH THE CLASSIFIER AND ADVERSARY NETWORKS
## creates the two networks, sets up Adam optimizers for both.
## Chooses loss functions: Binary cross-entropy with logits for the classifier (signal vs background)
##                     and Cross-entropy for the adversary (multi-class psum bin classification),
## assigns a mixing weight = 50 to control how strongly the adversary affects training.
classifier = Classifier(input_dim=X.shape[1], hidden_dims=args.hidden_dims, dropout=args.dropout, activation=args.activation)
#changed from 1 to 5
adversary = Adversary(input_dim=X.shape[1] if len(args.hidden_dims) == 0 else args.hidden_dims[-1])  # match classifier output logits

###########THESE ARE THEIR PARAMETERS
opt_clf = torch.optim.Adam(classifier.parameters(), lr=1e-3)
opt_adv = torch.optim.Adam(adversary.parameters(), lr=1e-3)

#########THIS IS THE LOSS FUNCTION FOR EACH
# Extracting the hidden node information to classifier to feed into the adversary
criterion_clf = torch.nn.BCEWithLogitsLoss()
criterion_adv = torch.nn.CrossEntropyLoss()
lambda_adv = 0#set this to 1 (BETTER ROC curves but leakages, if high better leakage cover but worse performance - to make both of them good, we add more layers of classifier as inputs to the adversary) INITIALLY WAS 50
#leakage: force neural network to invariant with respect to whatever the adversary is killing: rn mass, more leakage = classifier using mass to classify/predict

##ROC CURVE FOR EPOCH 0 - EPOCH 1
roc_snaps = {}

##helper functions to get true labels and scores on the test dataset
def get_scores(data_loader):
    """Run the classifier on the test set and return:
    - y_true array: the actual labels (0 or 1 - signal or background)
    - y_score_chunks array: the predicted probability of '1' for each event"""

    #Put the model in evaluation mode so it doesn't change its internal state
    classifier.eval()

    #Collect results in lists, then combine into big arrays
    y_true_chunks = []
    y_score_chunks = []

    #"no_grad" means to not track gradients as this is for testing
    with torch.no_grad():
        for xb, yb, _ in data_loader:
            #xb: batch of input features (shape: batch_size x num_features)
            #yb": true labels for that batch
            #"_": unused adversary label

            # 1) run the classifer: get raw outputs called "logits"
            logits, _ = classifier(xb)
            # 2) convert logits to probabilities via the sigmoid function
            probs  = torch.sigmoid(logits)
            # 3) move data to NumPy and store
            y_true_chunks.append(yb.numpy())
            y_score_chunks.append(probs.numpy())
    #return model to "training mode" so subsequent calls modify weights again
    classifier.train()

    #concatenate all chuncks to single arrays and return
    y_true  = np.concatenate(y_true_chunks)
    y_score = np.concatenate(y_score_chunks)
    return y_true, y_score

##helper function to take a snapshot
def take_roc_snapshot(stage_name, data_loader):
    """Compute test-set scores and store them in the global
       roc_snaps dict under the given key"""
    y_true, y_score = get_scores(data_loader)
    roc_snaps[stage_name] = (y_true, y_score)

##########WE NOW TRAIN OVER THE SAME TRAINING SET epoch MANY TIMES TO GET STABLE PERFORMANCE
## forward pass through classifier to get a logit per event, compute classification loss
## comparing to the true signal/background label, freeze the classifier (using .detach()) and pass
## its output through the adversary - applying gradient reversal internally, compute adversary loss on the
## true bin labels, train the classifier to both minimize its own loss and (because of gradient reversal)
## make the adversary's job harder, train the adversary to get better at its bin-classification task.


##prepare lists to record loss values over epochs
train_loss_list = [] #total training loss per epoch
validate_loss_list = []

classifier.eval() #evaluation mode - no weight changes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier.to(device)
adversary.to(device)

# resume training mode so weights updates
classifier.train()
adversary.train()

def calc_classifier_loss(data_loader):
    classifier.eval()
    loss = 0
    with torch.no_grad():
        for xb, yb, _ in data_loader:
            logits, _ = classifier(xb)
            loss += criterion_clf(logits, yb.float()).item()
    classifier.train()        
    return loss

print("Total events :", X.shape[0])

MAX_EVENTS_PER_FILE = 10000
for epoch in range(1, EPOCH+1):
    print('epoch:', epoch)
    #--------TRAINING & VALIDATION PASS ----------#
    epoch_loss_sum = num_batches = 0
    validate_loss_sum = num_val_batches = 0
    samples_seen = 0
    
    #make a counter until 10 k and exit
    for x, y, ycut in train_loader:  #have to do enumerate for mid roc skipped
        x, y, ycut = x.to(device), y.to(device), ycut.to(device)
            # Optional: Check for NaNs/Infs

        batch_size = x.size(0)
        # if taking this batch would exceed 10k, stop training for this epoch
        if samples_seen + batch_size > MAX_EVENTS_PER_FILE:
            print(f"Epoch {epoch}: reached {samples_seen} samples, stopping at 10k")
            break

        # Step 1: forward through classifier
         #ADV-ED COMMENTED OUT FOMR LOGITS = CLASSIFIER(X) TO:
        logits, coding = classifier(x)

        # Diagnostics for mismatch
        if coding.shape[0] != x.shape[0]:
            print("⚠️ Skipping batch: mismatch between input and coding shape")
            print(f"x.shape = {x.shape}")
            print(f"coding.shape = {coding.shape}")
            print("Input x (sample):", x[:5])  # print only first few for brevity
            print("Coding (sample):", coding[:5])
            # assert coding.shape[0] == x.shape[0], "Mismatch in batch size between x and coding"
            continue
        
        if coding.shape[0] != x.shape[0]:
            print("⚠️ Skipping batch: mismatch between input and output")
            continue
        # Step 2: classification loss
        loss_clf = criterion_clf(logits, y.float())

        epoch_loss_sum += loss_clf.item()
        num_batches += 1
        samples_seen += batch_size

        # Step 3: adversarial prediction (on logits or softmax)
        adv_input = coding.detach()
        dz_logits = adversary(adv_input, lambda_=lambda_adv) #feeds that single node to adversary - classifier output
        

        loss_adv = criterion_adv(dz_logits, ycut)

        # Step 5: combined loss (gradient flows through classifier only from clf_loss)
        total_loss = loss_clf + lambda_adv * loss_adv

        # Step 6: update classifier
        opt_clf.zero_grad()
        total_loss.backward(retain_graph=True)  # keep graph for adversary update
        opt_clf.step()

        # Step 7: update adversary separately
        opt_adv.zero_grad()
        loss_adv.backward()
        opt_adv.step()

    
    train_loss_list.append(epoch_loss_sum)
    validate_loss_list.append(calc_classifier_loss(val_loader))


plt.figure()
plt.plot(range(1, EPOCH+1), train_loss_list, marker='o', label='Total Training Loss')
plt.plot(range(1, EPOCH+1), validate_loss_list, marker='s', label='Total Validation Loss')
plt.xticks(range(1, EPOCH+1))
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss/Epoch")
plt.legend()
plt.savefig(f"{args.output_dir}/TOTAL_TRAINING_AND_VALIDATION_LOSS_PER_EPOCH.png")

def calculate_predictions(data_loader):
    y_true_signal = []
    y_pred_signal = []

    y_true_region = []
    y_pred_region_logits = []

    with torch.no_grad():
        for x_batch, y_signal, y_region in data_loader:
            logits_signal, coding = classifier(x_batch) # ADV-ED COMMENTED OUT - ADDED FEATURE
            probs_signal = torch.sigmoid(logits_signal)
            logits_region = adversary(coding.detach(), lambda_=1.0)

            y_true_signal.extend(y_signal.numpy())
            y_pred_signal.extend(probs_signal.numpy())

            y_true_region.extend(y_region.numpy())
            y_pred_region_logits.extend(logits_region.numpy())  # shape: (B, n_bins)

    return y_true_signal, y_pred_signal, y_true_region, y_pred_region_logits


pz_col = 12

signal_mask_train = (Y_train == 1)
bkg_mask_train = (Y_train == 0)

pz_signal_train = X_train[signal_mask_train, pz_col]
pz_bkg_train = X_train[bkg_mask_train, pz_col]

plt.figure()
plt.hist(pz_signal_train,
    bins=50,
    density=True,
    histtype='step',
    label='Signal (training)')
plt.xlabel('ele.track_.pz_ (GeV/c)')
plt.ylabel('Density')
plt.hist(pz_bkg_train,
        bins=50,
        density=True,
        histtype='step',
        label='Bkg (training)')
plt.xlabel('ele.track_.pz_ (GeV/c)')
plt.ylabel('Density')
plt.savefig("YUJINAS FIRST TASK TRAIN- Signal vs Background.png")
plt.close()
#### TEST

signal_mask_test = (Y_test == 1)
bkg_mask_test = (Y_test == 0)

pz_signal_test = X_test[signal_mask_test, pz_col]
pz_bkg_test = X_test[bkg_mask_test, pz_col]

plt.figure()
plt.hist(pz_signal_test,
    bins=50,
    density=True,
    histtype='step',
    label='Signal (training)')
plt.xlabel('ele.track_.pz_ (GeV/c)')
plt.ylabel('Density')
plt.hist(pz_bkg_test,
        bins=50,
        density=True,
        histtype='step',
        label='Bkg (training)')
plt.xlabel('ele.track_.pz_ (GeV/c)')
plt.ylabel('Density')
plt.savefig("YUJINAS FIRST TASK TEST - Signal vs Background.png")
plt.close()

print("Total events :", X.shape[0])
for idx, name in enumerate(flattened_names):
  train_vals = X_train[Y_train == 1, idx], X_train[Y_train == 0, idx] #mask and then idx means column for feature element (matrix composition: rows - event # (?), columns - feature/types like ele.pos, ele.pz)
  test_vals = X_test[Y_test == 1, idx], X_test[Y_test == 0, idx]
  #figure with two subplots
  fig, (ax_tr, ax_te) = plt.subplots(1, 2, figsize=(10,4), sharey=True)

  #train-set histograms
  ax_tr.hist(train_vals[1], bins=50, density=True, histtype='step', label ='Bkg(train)')
  ax_tr.hist(train_vals[0], bins=50, density=True, histtype='step', label ='Sig(train)')
  ax_tr.set_title(f'Train: {name}')
  ax_tr.set_xlabel(name)
  ax_tr.set_ylabel('Density')
  ax_tr.legend()

  #test-set histograms
  ax_te.hist(test_vals[1], bins=50, density=True, histtype='step', label ='Bkg(test)')
  ax_te.hist(test_vals[0], bins=50, density=True, histtype='step', label ='Sig(test)')
  ax_te.set_title(f'Test: {name}')
  ax_te.set_xlabel(name)
  ax_te.legend()

  fig.suptitle(f'Distribution of {name}', fontsize=15)
  fig.tight_layout()
  safe_name = name.replace('.', '_').replace('[', '_').replace(']', '')
  fig.savefig(f"Z(TO KEEP IT IN ONE PLACE): {safe_name}.png")
  plt.close()

############THIS PLOTS THE ROC CURVE################
def plot_roc(y_true, y_score, title):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    try:
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        helper=str(title)+"NoMassDiscrFarZ.png"
        plt.savefig(helper)
    except ValueError:
        print("One of the ROC Cuves Failes")

############THIS PLOTS SIGNIFICANCE, OR THE NUMBER OF SIGNAL EVENTS OVER THE NUMBER OF BACKGROUND################
def plot_sig(y_true, y_score, title):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    try:
        def helper(a,b):
            if a>=.00001:
                return b/np.sqrt(a)
            return 0
        sig = [helper(fpr[i],tpr[i]) for i in range(len(tpr))]
        roc_auc = max(sig)
        plt.figure()
        plt.plot(fpr, sig, lw=2, label=f'SIG = {roc_auc:.2f}')#, fpr = {FPR:.2f}, tpr =  {TPR:.2f}')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('Scan Parameter')
        plt.ylabel('Significance')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        helper=str(title)+"NoMassDiscrFarZ"+str(EPOCH)+".png"
        plt.savefig(helper)
    except ValueError:
        print("One of the ROC Cuves Failes")

##### THIS PLOTS THE ROC CURVES FOR THE CLASSIFIER #############
# Classifier ROC (signal vs background)
y_true_signal_val, y_pred_signal_val, _, _ = calculate_predictions(val_loader)
plot_roc(y_true_signal_val, y_pred_signal_val, f"{args.output_dir}/Classifier_ROC_Signal_vs_Background_Validation")
# plot_sig(y_true_signal_val, y_pred_signal_vanl, "Classifier_SIG_Signal_vs_Background_Validation")

# Classifier ROC (signal vs background)
y_true_signal_train, y_pred_signal_train, _, _ = calculate_predictions(train_loader)
plot_roc(y_true_signal_train, y_pred_signal_train, f"{args.output_dir}/Classifier_ROC_Signal_vs_Background_Train")
# plot_sig(y_true_signal, y_pred_signal, "Classifier_SIG_Signal_vs_BackgroundTrain")

y_true_signal_test, y_pred_signal_test, _, _ = calculate_predictions(test_loader)
plot_roc(y_true_signal_test, y_pred_signal_test, f"{args.output_dir}/Classifier_ROC_Signal_vs_Background_Test")

y_pred_test = (np.array(y_pred_signal_test) > 0.5).astype(int)
cm = confusion_matrix(y_true_signal_test, y_pred_test)

print("Confusion Matrix:")
print(cm)

# Calculate accuracy
accuracy = accuracy_score(y_true_signal_test, y_pred_test)
print(f"Accuracy: {accuracy}")

# Calculate precision
precision = precision_score(y_true_signal_test, y_pred_test)
print(f"Precision: {precision}")


logits_signal = []
logits_background = []

classifier.eval()
with torch.no_grad():
    for x_batch, y_batch, y_region in test_loader:
        logits, _ = classifier(x_batch)  # This should be the *pre-sigmoid* value
        logits = logits.squeeze()
        for logit, label in zip(logits, y_batch):
            if label == 1:  # Signal
                logits_signal.append(logit.item())
            else:          # Background
                logits_background.append(logit.item())

# Convert to numpy for plotting
logits_signal = np.array(logits_signal)
logits_background = np.array(logits_background)

# Plot
plt.clf()
## visualizes how separable the pre-sigmoid outputs are for signal vs. background
plt.hist(logits_signal, bins=50, alpha=0.5, label='Signal', density=True, histtype='step')
plt.hist(logits_background, bins=50, alpha=0.5, label='Background', density=True, histtype='step')
plt.xlabel('Classifier Output (Pre-Sigmoid Logit)')
plt.ylabel('Density')
plt.legend()
plt.title('Logit Distribution Before Sigmoid')
plt.grid(True)
plt.savefig("SigmoidBingBongNoMassDiscrFarZ.png")
