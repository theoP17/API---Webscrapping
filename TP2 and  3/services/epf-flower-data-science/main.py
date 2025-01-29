import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from src.api.routes.data import router as data_router
from src.api.routes.parameters import router as parameters_router

from src.app import get_application

app = get_application()

@app.get("/")
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

app.include_router(data_router)
app.include_router(parameters_router)

if __name__ == "__main__":
    uvicorn.run("main:app", debug=True, reload=True, port=8080)
