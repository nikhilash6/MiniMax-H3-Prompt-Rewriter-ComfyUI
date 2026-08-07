import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_ID = "MiniMaxH3PromptRewriter";
const OPEN_URL = "/minimax_h3_rewriter/open_model_list";
const BUTTON_LABEL = "Open model list";

async function openModelList() {
    let response;
    try {
        response = await api.fetchApi(OPEN_URL, { method: "POST" });
    } catch (error) {
        alert(`Could not reach the ComfyUI server: ${error}`);
        return;
    }

    let payload = {};
    try {
        payload = await response.json();
    } catch (error) {
        payload = {};
    }

    if (!response.ok || !payload.ok) {
        alert(
            "Could not open the model list.\n\n" +
                (payload.error || `HTTP ${response.status}`) +
                "\n\nEdit it by hand instead:\n" +
                (payload.path || "ComfyUI/user/minimax_h3_rewriter/models.json")
        );
        return;
    }

    console.log(`[MiniMax-H3 Prompt Rewriter] opened ${payload.path}`);
}

app.registerExtension({
    name: "minimax_h3_rewriter.model_list",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_ID) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const widget = this.addWidget("button", BUTTON_LABEL, null, openModelList);
            widget.serialize = false;
            widget.tooltip =
                "Opens models.json in the ComfyUI user directory. Add any Qwen3.6-27B " +
                "repacking there, then refresh the browser to see it in the dropdown.";
            return result;
        };
    },
});
