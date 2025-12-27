import logging

from lm_eval.models.huggingface import HFLM

logger = logging.getLogger(__name__)

class LMEvalWrapper(HFLM):
    def __init__(
            self,
            model,
            tokenizer,
            batch_size: int,
            device: str = "cuda:0",
            dtype: str = "float16",
    ) -> None:
        super().__init__(
            pretrained=f"facebook/opt-125m",
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            max_batch_size=64,
            trust_remote_code=True,
        )

        self._model.to("cpu")
        self._model = None
        # call GC
        import gc
        gc.collect()

        import torch
        torch.cuda.empty_cache()
        model = model.half()
        # check function exists
        if hasattr(model, "move_to_parallel"):
            model.move_to_parallel()
        self._model = model