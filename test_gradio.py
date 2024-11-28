import gradio as gr

def show_image():
    return "/home/mdnikolaev/darudenkov/InfEdit/nsfw.png"  

# Создаем интерфейс
with gr.Blocks() as demo:
    gr.Image(show_image)  

#https://www.gradio.app/main/docs/gradio/blocks
# server_port
demo.launch(server_name="0.0.0.0")