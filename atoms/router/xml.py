import os

from config import settings
from fastapi import APIRouter, Request, Response
from jinja2 import Environment, FileSystemLoader

router = APIRouter()

template_env = Environment(
    loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "..", "templates"))
)


@router.post("/twilio/xml")
async def twilio_xml(request: Request):
    """Serve TwiML for Twilio calls."""
    template = template_env.get_template("twilio_ws.xml")
    xml_content = template.render()

    return Response(content=xml_content, media_type="application/xml")


@router.post("/plivo/xml")
async def plivo_xml(request: Request):
    """Serve XML for Plivo calls."""
    template = template_env.get_template("plivo_ws.xml")
    xml_content = template.render(server_base_url=settings.server_base_url)

    return Response(content=xml_content, media_type="application/xml")


@router.get("/plivo/transfer")
async def plivo_transfer(request: Request):
    """Serve transfer XML for Plivo calls."""
    template = template_env.get_template("plivo_transfer.xml")
    xml_content = template.render()

    return Response(content=xml_content, media_type="application/xml")
