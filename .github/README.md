# PlayOn

Backend for PlayOn — DivaLopers AppCon Entry

## Get started

1. Clone the repository:

   ```bash
   git clone git@github.com:LowisWano/playon.git
   ```

2. Install dependencies

   ```bash
   python -m venv venv # virtualenv for dependencies
   .venv\bin\activate # activate venv
   pip install -r requirements.txt
   cd ./db
   prisma generate
   ```

3. Start the app

   ```bash
    uvicorn main:app --reload # hot reload
    http://<host>/docs # for swagger 
   ```


## Contribution Conventions

Here's an example commit flow with git:

```bash
# sync latest code from the remote repository
git pull
# create a new branch based on the feature you want to work on
git checkout -b <new_branch>
# after making some changes, add and commit your work
git add .
git commit -m "category: do something"
# push your changes and make a pull request on GitHub afterwards so that I can review them
git push origin HEAD
```
Feel free to ask questions for conflicts.

