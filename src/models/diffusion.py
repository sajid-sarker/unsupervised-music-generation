from __future__ import annotations


class DiffusionModelPlaceholder:
	def __init__(self) -> None:
		self.message = "Diffusion model is outside the current project requirements (Tasks 1-3)."

	def __repr__(self) -> str:
		return self.message

