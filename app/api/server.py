from fastapi import FastAPI


from app.api.rl import router as rl_router
from app.api.baseline import router as baseline_router       
from app.api.rl_signals import router as rl_signals_router   
from app.api.backtest import router as backtest_router  
from app.api.data import router as data_router
from app.api.explain import router as explain_router
app = FastAPI(title="TradeMind Backend ")

@app.get("/healthz", tags=["health"])
def healthz():
    return {"ok": True}


app.include_router(rl_router)
app.include_router(baseline_router)     
app.include_router(rl_signals_router)    
app.include_router(backtest_router)   
app.include_router(data_router)
app.include_router(explain_router)



