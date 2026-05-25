# Dublin Temperature Predictor

Small ML app that fetches the last N days of Dublin's daily mean
temperatures and predicts the next day's temperature. Used as the
worked example for the MLOps pipeline assessment (data acquisition →
training → deployment via Flask + Docker + Kubernetes + ArgoCD + GitHub Actions).

## Stack

- Python 3.11
- Open-Meteo forecast API (`past_days` window, no key required) for historical data
- pandas / numpy for data wrangling
- scikit-learn Ridge regression with lag and seasonal features
- joblib for model persistence
- Flask + gunicorn for serving
- Docker (two images: trainer + API), pushed to GHCR
- GitHub Actions for CI / Train / CD / Continuous Training
- Kustomize for image-tag management in `k8s/`
- ArgoCD running in a home-server Kubernetes cluster, syncing `k8s/` from this repo

## Project layout

```
.
├── main.py                         # CLI: fetch → train → predict
├── Dockerfile.train                # Trainer image
├── Dockerfile.api                  # Serving image (gunicorn + Flask)
├── requirements-train.txt
├── requirements-serve.txt
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── kustomization.yaml          # image tag rewritten by cd.yml
├── argocd/
│   └── application.yaml            # ArgoCD Application CR
├── .github/workflows/
│   ├── ci.yml                      # lint + pytest + docker build (PRs)
│   ├── train.yml                   # data acquisition + train + test
│   ├── cd.yml                      # build image + bump tag in k8s/ + commit
│   └── ct.yml                      # scheduled retrain → redeploy
├── src/temp_predictor/
│   ├── data.py                     # Open-Meteo client (Dublin)
│   ├── features.py                 # lag / rolling / seasonal features
│   ├── train.py                    # Ridge pipeline + time-aware split
│   ├── predict.py                  # load model + forecast next day
│   └── api.py                      # Flask API
└── tests/
    ├── test_features.py
    └── test_api.py
```

## Branching strategy (GitHub Flow)

- `main` is protected and always deployable.
- Work happens on `feature/*` or `fix/*` branches.
- Pull requests into `main` trigger **CI** (`ci.yml`): ruff, pytest, and
  `docker build` of both images (no push).
- Merge to `main` triggers **Train** (`train.yml`); a successful train run
  triggers **CD** (`cd.yml`) which builds the API image, pushes it to GHCR,
  and writes the new tag into `k8s/kustomization.yaml` on `main`.
- **ArgoCD**, running on the home-server cluster, polls `main`, detects the
  manifest change, and rolls out the new image automatically (auto-sync +
  self-heal).

The deploy pipeline is GitOps: a successful CI run produces a *commit*,
not a `kubectl apply`. The cluster pulls; it isn't pushed to.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-serve.txt

# Train and predict from the CLI
python main.py --days 90

# Serve predictions over HTTP (dev server)
python -m src.temp_predictor.api
# or production-style
gunicorn -w 2 -b 0.0.0.0:8000 'src.temp_predictor.api:app'
```

Datasets are written to `data/` and models to `models/ridge.joblib`.
Both directories are gitignored.

## Run with Docker

```bash
# Train inside the trainer container — outputs models/ridge.joblib
docker build -f Dockerfile.train -t temp-predictor-trainer .
docker run --rm \
  -v "$PWD/models:/app/models" \
  -v "$PWD/data:/app/data" \
  temp-predictor-trainer --days 90

# Build the API image with the model baked in
docker build -f Dockerfile.api -t temp-predictor-api .
docker run --rm -p 8000:8000 temp-predictor-api
curl http://localhost:8000/health
curl "http://localhost:8000/predict?days=90"
```

## Home-server deployment (one-time setup)

These steps run once on the home-server box that hosts the Kubernetes
cluster. After they're done, all subsequent deploys happen automatically
via GitHub Actions → GHCR → ArgoCD.

### 1. Cluster

Either Kind (lighter, fine for the demo) or K3s (more "production"). Pick one.

```bash
# Option A: Kind
kind create cluster --name temp-predictor

# Option B: K3s
curl -sfL https://get.k3s.io | sh -
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config && sudo chown $(id -u) ~/.kube/config
```

### 2. ArgoCD

```bash
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
kubectl -n argocd rollout status deployment/argocd-server --timeout=180s

# Expose the UI on localhost:8080 (or set up an Ingress if you prefer)
kubectl port-forward -n argocd svc/argocd-server 8080:443 &

# Initial admin password (the username is `admin`):
kubectl -n argocd get secret argocd-initial-admin-secret \
  -o jsonpath='{.data.password}' | base64 -d; echo
```

### 3. Register the app

Edit `argocd/application.yaml` and replace `REPLACE_ME` with your GitHub
account (or the full repo URL), then:

```bash
kubectl apply -f argocd/application.yaml
```

### 4. Make the GHCR image public (one-time)

After the first successful `cd.yml` run, the API image lands at
`ghcr.io/<you>/<repo>/temp-predictor-api`. On GitHub:

- Your profile → Packages → `temp-predictor-api` → Package settings →
  Change visibility → **Public**.

This means the cluster can pull anonymously and no `imagePullSecrets`
are needed in `k8s/deployment.yaml`. (If you'd rather keep it private,
add a pull secret to the `default` namespace and reference it in the
deployment.)

That's the entire setup. From now on, every push to `main` that produces
a new model causes ArgoCD to roll out the new image within a couple of
minutes.

## Branch protection note

`cd.yml` needs to push the manifest-bump commit to `main`, which is
protected by a ruleset requiring PR review. The bypass list on personal
repositories only surfaces roles (Admin / Maintain / Write) and Deploy
keys — it does **not** expose `github-actions[bot]` as a selectable
actor. To let CD push automated tag bumps without disabling protection
for human commits, we authenticate the push as the admin user via a
fine-grained Personal Access Token.

### One-time setup

1. **Generate a fine-grained PAT** — GitHub → Settings (user) →
   Developer settings → Personal access tokens → Fine-grained tokens →
   Generate new token.
   - Repository access: only this repo.
   - Permissions → Repository → **Contents: Read and write**
     (Metadata: Read-only is auto-required).
2. **Add the token as a repo secret** — Repo → Settings → Secrets and
   variables → Actions → New repository secret, name `CD_PUSH_TOKEN`.
3. **Confirm the ruleset bypass list** includes `Repository admin Role`
   (mode `always`). The PAT inherits the admin's bypass, so the push
   from the CD workflow is accepted while human pushes to `main` still
   require a PR.

The relevant line in `.github/workflows/cd.yml`:

```yaml
- uses: actions/checkout@v4
  with:
    token: ${{ secrets.CD_PUSH_TOKEN }}
    fetch-depth: 0
```

This keeps the spirit of "humans go through PRs" while allowing the
post-train manifest bump (which is not a code change anyone would
review by hand) to flow through automatically.

### Update 2026-05-25

Switched CD from the default `GITHUB_TOKEN` to a fine-grained PAT
(`CD_PUSH_TOKEN`) after the protected-branch push was rejected with
`GH006: Protected branch update failed for refs/heads/main`. Root cause:
ruleset bypass on personal repos doesn't expose the GitHub Actions app
as a selectable actor, so the bot's push was treated as a regular
unreviewed write to `main`.

## API

| Method | Path        | Description                                          |
| ------ | ----------- | ---------------------------------------------------- |
| GET    | `/health`   | Liveness/readiness probe (returns version).          |
| GET    | `/predict`  | `?days=90` → next-day Dublin forecast.               |
| POST   | `/predict`  | JSON body with custom `history` → next-day forecast. |

## Pipeline stages → workflows

| Stage                             | Workflow / Tool                                      |
| --------------------------------- | ---------------------------------------------------- |
| Continuous Integration            | `.github/workflows/ci.yml`                           |
| Data Acquisition & Preprocessing  | `train.yml` (inside the trainer container)           |
| Model Training & Testing          | `train.yml`                                          |
| Continuous Delivery               | `cd.yml` (build + push + manifest commit)            |
| Model Deployment                  | ArgoCD (pulls `k8s/` from this repo, applies to cluster) |
| Continuous Training               | `ct.yml` (cron + reuse of `train.yml` + `cd.yml`)    |
