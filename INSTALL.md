# General animaldet installation guide

Animaldet was built as a containerized application which can run locally
or at any cloud which support containers

# Local deployment

First of all with the repo cloned you must execute 
```bash
git lfs pull
```

To get models and other useful information to deploy your model.

You should have `docker` installed.

After that you can execute

```bash
make build
docker run -p 8000:8000 animaldet:latest
```

Then when opening in your browser `localhost:8000` you must see the application running. 
![Main view](static/img/main_view.png)

For user guide about how to use Animaldet you can check the [next link](USER_GUIDE.md)
# Cloud deployment

Animaldet included deployment platform is `AWS ECS`, with the next architecture:

![Architecture](static/img/architecture.png)


We included all the required infrastructure as a code at [infra/aws](infra/aws), there we use `terraform`.

In order to deploy the application you must configure your aws credentials, more information at [aws cli configuration documentation](https://docs.aws.amazon.com/cli/v1/userguide/cli-configure-files.html)

Commands to deploy the application 

```bash
make deploy
```

It will execute `docker build`, then `terraform apply` which is going to:

1. Build docker image locally
2. Login you to ECS registry
3. Push the image to ECS
4. Create clusters, task definitions, policies and load balancer to get the application working.

