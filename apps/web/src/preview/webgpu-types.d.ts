/**
 * Minimal ambient WebGPU surface
 * ==============================
 *
 * Hand-written on purpose. The project ships **zero runtime dependencies** and
 * `@webgpu/types` is a dev dependency we do not take; everything below is only the
 * subset the preview renderer actually calls, transcribed from the WebGPU IDL.
 *
 * Two consequences worth knowing:
 *   • This file is the only place a WebGPU API assumption lives. If a member is
 *     missing here, the fix is to add it — not to cast to `any` at the call site.
 *   • `GPUError` is intentionally NOT declared: it already exists in TypeScript's
 *     `lib.dom.d.ts`, and redeclaring it would be a merge conflict.
 *
 * The browser is the authority on whether a call is legal; this just keeps `tsc`
 * honest about the shapes.
 */

type GPUTextureFormat = string;
type GPUBufferUsageFlags = number;
type GPUTextureUsageFlags = number;
type GPUMapModeFlags = number;
type GPUErrorFilter = 'validation' | 'out-of-memory' | 'internal';
type GPUAddressMode = 'clamp-to-edge' | 'repeat' | 'mirror-repeat';
type GPUFilterMode = 'nearest' | 'linear';
type GPUMipmapFilterMode = 'nearest' | 'linear';

interface GPURequestAdapterOptions {
  powerPreference?: 'low-power' | 'high-performance';
  forceFallbackAdapter?: boolean;
}

interface GPUDeviceDescriptor {
  label?: string;
  requiredFeatures?: string[];
  requiredLimits?: Record<string, number>;
}

interface GPUSupportedFeatures {
  has(feature: string): boolean;
}

interface GPUAdapterInfo {
  readonly vendor: string;
  readonly architecture: string;
  readonly device: string;
  readonly description: string;
}

interface GPU {
  requestAdapter(options?: GPURequestAdapterOptions): Promise<GPUAdapter | null>;
  getPreferredCanvasFormat(): GPUTextureFormat;
}

interface GPUAdapter {
  readonly features: GPUSupportedFeatures;
  readonly info?: GPUAdapterInfo;
  requestDevice(descriptor?: GPUDeviceDescriptor): Promise<GPUDevice>;
}

interface GPUDeviceLostInfo {
  readonly reason: string;
  readonly message: string;
}

interface GPUCompilationMessage {
  readonly message: string;
  readonly type: 'error' | 'warning' | 'info';
  readonly lineNum: number;
  readonly linePos: number;
  readonly offset: number;
  readonly length: number;
}

interface GPUCompilationInfo {
  readonly messages: readonly GPUCompilationMessage[];
}

interface GPUShaderModuleDescriptor {
  label?: string;
  code: string;
}

interface GPUShaderModule {
  getCompilationInfo(): Promise<GPUCompilationInfo>;
}

interface GPUBufferDescriptor {
  label?: string;
  size: number;
  usage: GPUBufferUsageFlags;
  mappedAtCreation?: boolean;
}

interface GPUBuffer {
  readonly size: number;
  readonly usage: GPUBufferUsageFlags;
  readonly mapState: 'unmapped' | 'pending' | 'mapped';
  getMappedRange(offset?: number, size?: number): ArrayBuffer;
  unmap(): void;
  mapAsync(mode: GPUMapModeFlags, offset?: number, size?: number): Promise<void>;
  destroy(): void;
}

type GPUExtent3D =
  | [number, number, number]
  | [number, number]
  | { width: number; height?: number; depthOrArrayLayers?: number };

interface GPUTextureDescriptor {
  label?: string;
  size: GPUExtent3D;
  mipLevelCount?: number;
  sampleCount?: number;
  dimension?: '1d' | '2d' | '3d';
  format: GPUTextureFormat;
  usage: GPUTextureUsageFlags;
}

interface GPUTextureViewDescriptor {
  label?: string;
  format?: GPUTextureFormat;
  dimension?: '1d' | '2d' | '2d-array' | 'cube' | 'cube-array' | '3d';
  aspect?: 'all' | 'stencil-only' | 'depth-only';
  baseMipLevel?: number;
  mipLevelCount?: number;
  baseArrayLayer?: number;
  arrayLayerCount?: number;
}

interface GPUTextureView {
  readonly label?: string;
}

interface GPUTexture {
  readonly width: number;
  readonly height: number;
  createView(descriptor?: GPUTextureViewDescriptor): GPUTextureView;
  destroy(): void;
}

interface GPUSamplerDescriptor {
  label?: string;
  addressModeU?: GPUAddressMode;
  addressModeV?: GPUAddressMode;
  addressModeW?: GPUAddressMode;
  magFilter?: GPUFilterMode;
  minFilter?: GPUFilterMode;
  mipmapFilter?: GPUMipmapFilterMode;
  lodMinClamp?: number;
  lodMaxClamp?: number;
  compare?: string;
  maxAnisotropy?: number;
}

interface GPUSampler {
  readonly label?: string;
}

interface GPUQueue {
  writeBuffer(
    buffer: GPUBuffer,
    bufferOffset: number,
    data: BufferSource,
    dataOffset?: number,
    size?: number,
  ): void;
  submit(commandBuffers: GPUCommandBuffer[]): void;
}

interface GPUCommandBuffer {
  readonly label?: string;
}

interface GPUCommandEncoderDescriptor {
  label?: string;
}

interface GPUCommandBufferDescriptor {
  label?: string;
}

interface GPUColor {
  r: number;
  g: number;
  b: number;
  a: number;
}

interface GPURenderPassColorAttachment {
  view: GPUTextureView;
  resolveTarget?: GPUTextureView;
  clearValue?: GPUColor;
  loadOp: 'load' | 'clear';
  storeOp: 'store' | 'discard';
}

interface GPURenderPassDescriptor {
  label?: string;
  colorAttachments: GPURenderPassColorAttachment[];
}

interface GPURenderPassEncoder {
  setPipeline(pipeline: GPURenderPipeline): void;
  setBindGroup(index: number, bindGroup: GPUBindGroup, dynamicOffsets?: number[]): void;
  draw(vertexCount: number, instanceCount?: number, firstVertex?: number, firstInstance?: number): void;
  end(): void;
}

interface GPUCommandEncoder {
  beginRenderPass(descriptor: GPURenderPassDescriptor): GPURenderPassEncoder;
  finish(descriptor?: GPUCommandBufferDescriptor): GPUCommandBuffer;
}

interface GPUBindGroupEntry {
  binding: number;
  resource: GPUBindingResource;
}

type GPUBindingResource =
  | GPUSampler
  | GPUTextureView
  | { buffer: GPUBuffer; offset?: number; size?: number };

interface GPUBindGroupLayout {
  readonly label?: string;
}

interface GPUBindGroupDescriptor {
  label?: string;
  layout: GPUBindGroupLayout;
  entries: GPUBindGroupEntry[];
}

interface GPUBindGroup {
  readonly label?: string;
}

interface GPUPipelineLayout {
  readonly label?: string;
}

interface GPUColorTargetState {
  format: GPUTextureFormat;
  blend?: unknown;
  writeMask?: number;
}

interface GPUVertexState {
  module: GPUShaderModule;
  entryPoint?: string;
  constants?: Record<string, number>;
  buffers?: readonly unknown[];
}

interface GPUFragmentState {
  module: GPUShaderModule;
  entryPoint?: string;
  constants?: Record<string, number>;
  targets: readonly (GPUColorTargetState | null)[];
}

interface GPUPrimitiveState {
  topology?: 'point-list' | 'line-list' | 'line-strip' | 'triangle-list' | 'triangle-strip';
  stripIndexFormat?: GPUTextureFormat;
  frontFace?: 'ccw' | 'cw';
  cullMode?: 'none' | 'front' | 'back';
}

interface GPURenderPipelineDescriptor {
  label?: string;
  layout: GPUPipelineLayout | 'auto';
  vertex: GPUVertexState;
  fragment?: GPUFragmentState;
  primitive?: GPUPrimitiveState;
  depthStencil?: unknown;
  multisample?: unknown;
}

interface GPURenderPipeline {
  readonly label?: string;
  getBindGroupLayout(index: number): GPUBindGroupLayout;
}

interface GPUCanvasConfiguration {
  device: GPUDevice;
  format?: GPUTextureFormat;
  usage?: GPUTextureUsageFlags;
  alphaMode?: 'opaque' | 'premultiplied';
  colorSpace?: 'srgb' | 'display-p3';
}

interface GPUCanvasContext {
  readonly canvas: HTMLCanvasElement;
  configure(configuration: GPUCanvasConfiguration): void;
  unconfigure(): void;
  getCurrentTexture(): GPUTexture;
}

interface GPUDevice {
  readonly label?: string;
  readonly features: GPUSupportedFeatures;
  readonly lost: Promise<GPUDeviceLostInfo>;
  readonly queue: GPUQueue;
  createShaderModule(descriptor: GPUShaderModuleDescriptor): GPUShaderModule;
  createRenderPipeline(descriptor: GPURenderPipelineDescriptor): GPURenderPipeline;
  createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer;
  createTexture(descriptor: GPUTextureDescriptor): GPUTexture;
  createSampler(descriptor?: GPUSamplerDescriptor): GPUSampler;
  createBindGroup(descriptor: GPUBindGroupDescriptor): GPUBindGroup;
  createCommandEncoder(descriptor?: GPUCommandEncoderDescriptor): GPUCommandEncoder;
  pushErrorScope(filter: GPUErrorFilter): void;
  popErrorScope(): Promise<GPUError | null>;
  destroy(): void;
}

interface Navigator {
  /** Present only in secure contexts with WebGPU enabled; always feature-detect. */
  readonly gpu?: GPU;
}
