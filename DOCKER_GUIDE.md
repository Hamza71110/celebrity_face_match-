# Docker Guide — Celebrity Face Matcher

This document explains how to containerize and run the **Celebrity Face Matcher**
Streamlit app, and which screenshots to capture for submission.

---

## 1. Files added for containerization

| File              | Purpose                                                              |
|-------------------|---------------------------------------------------------------------|
| `Dockerfile`      | Recipe that builds the image (base image, deps, code, start command)|
| `requirements.txt`| Exact Python dependencies (pinned versions) installed in the image  |
| `.dockerignore`   | Excludes `.venv/`, IDE files, and scratch data from the build       |

> The app also needs `embedding.pkl`, `filenames.pkl`, and the `data/` folder at
> runtime — these are **kept** in the image (not ignored).

---

## 2. Prerequisites

1. Install **Docker Desktop**: <https://www.docker.com/products/docker-desktop/>
2. Start Docker Desktop and wait until it says **"Engine running"**.
3. Verify from a terminal:
   ```bash
   docker --version
   ```

---

## 3. Build the image

From the project folder (the one containing the `Dockerfile`):

```bash
docker build -t final-project:v1 .
```

- `-t final-project:v1` names the image `final-project` with tag `v1`.
- The trailing `.` is the **build context** (current directory).
- First build takes several minutes (TensorFlow is a large download). Later
  builds are faster because Docker caches the dependency layer.

📸 **Screenshot 1 — Docker build:** capture the terminal showing the build
finishing with `naming to docker.io/library/final-project:v1` (or
`Successfully built ...`).

Confirm the image exists:

```bash
docker images
```

📸 **Screenshot 2 — Image list:** the row for `final-project   v1`.

---

## 4. Run the container

```bash
docker run -p 8000:8000 final-project:v1
```

- `-p 8000:8000` maps host port 8000 → container port 8000.
- Streamlit prints a line like `You can now view your Streamlit app ... :8000`.

Open a browser at **<http://localhost:8000>** and upload a face photo.

📸 **Screenshot 3 — Running container:** the terminal showing the Streamlit
startup log **and/or** the app open in the browser with a celebrity match result.

To stop the container: press `Ctrl+C`, or in another terminal:
```bash
docker ps            # find the CONTAINER ID
docker stop <ID>
```

### Optional: run detached (in the background)
```bash
docker run -d -p 8000:8000 --name face-match final-project:v1
docker logs -f face-match      # follow logs
docker stop face-match         # stop
```

---

## 5. Deliverables checklist

- [x] `Dockerfile`
- [x] `requirements.txt`
- [ ] Docker **build** screenshot (Screenshot 1)
- [ ] Running **container** screenshot (Screenshot 3)
- [x] Reference document (this file)

---

## 6. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `docker: command not found` | Docker Desktop not installed or not started. |
| Port 8000 already in use | Use another host port: `docker run -p 8080:8000 final-project:v1` → open `localhost:8080`. |
| `ImportError: libGL.so.1` | The `libgl1-mesa-glx` system package (already in the Dockerfile) provides this — rebuild without cache: `docker build --no-cache -t final-project:v1 .` |
| Build fails compiling a package | Add build tools to the Dockerfile: `apt-get install -y build-essential` in the system-libs step, then rebuild. |
| Matched celebrity image doesn't show | Ensure the `data/` folder was copied into the image (it is not in `.dockerignore`). |
| Very large image / slow build | Normal — TensorFlow 2.3.1 + model artifacts are heavy. `docker system prune` frees space from old builds. |

---

## 7. Handy Docker commands

```bash
docker images                 # list images
docker ps                     # list running containers
docker ps -a                  # list all containers (incl. stopped)
docker stop <id> / rm <id>    # stop / remove a container
docker rmi final-project:v1   # remove the image
docker system prune           # clean up dangling data
docker exec -it <id> bash     # open a shell inside a running container
```

---

## 8. Learn Docker — references

- **Official Get Started guide:** <https://docs.docker.com/get-started/>
- **Dockerfile reference:** <https://docs.docker.com/reference/dockerfile/>
- **Build best practices:** <https://docs.docker.com/build/building/best-practices/>
- **`docker run` reference:** <https://docs.docker.com/reference/cli/docker/container/run/>
- **Streamlit in Docker (official):** <https://docs.streamlit.io/deploy/tutorials/docker>
- **Docker 101 tutorial (interactive):** <https://www.docker.com/101-tutorial/>
- **Play with Docker (free sandbox):** <https://labs.play-with-docker.com/>
