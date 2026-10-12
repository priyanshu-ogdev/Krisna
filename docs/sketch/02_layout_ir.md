# Layout intermediate representation

The new primary Sketch output should be a typed, renderable UI layout IR, not pixels.

```json
{
  "version": "krisna.layout.v1",
  "canvas": {"width": 1024, "height": 1024},
  "nodes": [
    {
      "id": "hero",
      "type": "container|text|image|button|input|list|card|icon|divider",
      "bbox": [x, y, width, height],
      "parent": null,
      "z_index": 0,
      "style_tokens": {},
      "content_ref": null,
      "locked": false
    }
  ],
  "relations": [{"source": "a", "relation": "below|align_x|align_y|inside|near", "target": "b"}]
}
```

Coordinates are normalized to [0,1]. The IR must be canonicalized before training and deterministically rendered into a wireframe/control image.

## Hard constraints

- valid node types;
- bounds inside canvas;
- parent-child containment;
- no impossible negative sizes;
- locked nodes unchanged;
- relation consistency;
- maximum node count;
- deterministic ordering.

The renderer, not the generative model, is the authority for legality.
