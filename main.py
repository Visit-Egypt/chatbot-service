from cmath import log
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from loguru import logger
from starlette.exceptions import HTTPException
from starlette.middleware.cors import CORSMiddleware


from config import (
    API_PREFIX,
    DEBUG,
)
from errors.http_error import http_error_handler
from errors.validation_error import (
    http422_error_handler,
)

from chatbot import router



def get_application() -> FastAPI:
    application = FastAPI(title="ChatBot Service", debug=DEBUG)

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.add_exception_handler(HTTPException, http_error_handler)
    application.add_exception_handler(RequestValidationError, http422_error_handler)


    application.include_router(router, prefix=API_PREFIX)

    return application


app = get_application()