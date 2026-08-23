import React, { useState, useCallback, useMemo } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  Handle,
  Position,
  MarkerType,
  getBezierPath,
} from 'reactflow';
import 'reactflow/dist/style.css';
import dagre from 'dagre';

export const PERSPECTIVE_COUNTRY_CODES = {
  Botswana: 'BW',
  Kenya: 'KE',
  'South Africa': 'ZA',
  Zambia: 'ZM',
  Zimbabwe: 'ZW',
};

export function buildPerspectiveRequest(payload, selectedCountry) {
  const country = selectedCountry || 'Zimbabwe';
  return {
    ...payload,
    perspective_country: country,
    perspective_country_code: PERSPECTIVE_COUNTRY_CODES[country] || '',
  };
}

// ============================================================
// INSTALLATION NOTE:
// npm install reactflow dagre
// Ensure Tailwind CSS is configured in your Vite project.
// Replace the `reasoningData` object below with:
//   import reasoningData from './reasoning_OPP-003.json';
// ============================================================

// ============================================
// 1. DATA (Replace with your JSON import)
// ============================================
const reasoningData = {
  metrics: {
    total_vault_files_scanned: 1247,
    nodes_extracted: 89,
    map_chunks_processed: 342,
    estimated_manual_hours_saved: 45.5,
  },
  convergence_flow: {
    tier_1_anchors: [
      "Regulatory Filing OPP-003",
      "Channel Partner Agreement",
      "Intermediary Compliance Matrix",
      "Single-Source Justification Memo",
    ],
    tier_2_processing_chunks: [
      "Integrity Validation Protocol",
      "Regulatory Requirement Mapping",
      "Channel Risk Assessment",
      "Intermediary Model Analysis",
    ],
    tier_3_synthesis_logic:
      "The convergence analysis of Tier 1 source anchors against Tier 2 processing chunks reveals a consistent pattern: all regulatory pathways mandate channel integrity verification prior to intermediary designation. The OPP-003 filing explicitly requires single-source justification when multi-channel alternatives introduce compliance fragmentation. By synthesizing the compliance matrix with the partner agreement, the engine determines that a unified intermediary model satisfies both regulatory stringency and operational efficiency. This justifies the single-source intermediary architecture as the optimal execution roadmap, eliminating an estimated 45.5 hours of manual cross-reference and validation work.",
  },
  compiled_lineage_traces: [
    {
      source_node: "Regulatory Filing OPP-003",
      target_concept: "Integrity Validation Protocol",
      relationship_type: "MANDATES",
      extracted_fact:
        "OPP-003 requires pre-validation of all channel integrity checkpoints before intermediary approval.",
      logic_justification:
        "Section 4.2 of the filing establishes mandatory integrity gates that must be satisfied prior to any downstream processing.",
    },
    {
      source_node: "Regulatory Filing OPP-003",
      target_concept: "Regulatory Requirement Mapping",
      relationship_type: "DEFINES",
      extracted_fact: "The filing maps 14 distinct regulatory requirements to channel operations.",
      logic_justification:
        "Cross-referencing the requirement index with operational clauses yields a direct mapping without ambiguity.",
    },
    {
      source_node: "Channel Partner Agreement",
      target_concept: "Channel Risk Assessment",
      relationship_type: "INFORMS",
      extracted_fact: "Partner liability caps are directly correlated with channel integrity scores.",
      logic_justification:
        "Clause 7.3 links financial exposure to integrity metrics, creating a risk-adjusted model.",
    },
    {
      source_node: "Channel Partner Agreement",
      target_concept: "Intermediary Model Analysis",
      relationship_type: "CONSTRAINS",
      extracted_fact:
        "The agreement restricts intermediary selection to entities with active compliance certification.",
      logic_justification:
        "Certification requirements in Annex B filter the candidate pool to pre-qualified entities.",
    },
    {
      source_node: "Intermediary Compliance Matrix",
      target_concept: "Integrity Validation Protocol",
      relationship_type: "VALIDATES",
      extracted_fact: "The matrix provides the scoring rubric for integrity validation.",
      logic_justification:
        "Each matrix dimension corresponds to a validation checkpoint defined in the protocol.",
    },
    {
      source_node: "Intermediary Compliance Matrix",
      target_concept: "Regulatory Requirement Mapping",
      relationship_type: "SATISFIES",
      extracted_fact: "Matrix coverage spans all 14 mapped regulatory requirements.",
      logic_justification: "Completeness analysis shows 100% requirement coverage with zero gaps.",
    },
    {
      source_node: "Single-Source Justification Memo",
      target_concept: "Intermediary Model Analysis",
      relationship_type: "JUSTIFIES",
      extracted_fact: "The memo provides the business case for consolidating to a single intermediary.",
      logic_justification:
        "Cost-benefit analysis and risk aggregation data support the single-source conclusion.",
    },
    {
      source_node: "Single-Source Justification Memo",
      target_concept: "Channel Risk Assessment",
      relationship_type: "MITIGATES",
      extracted_fact: "Consolidation reduces fragmentation risk by 68%.",
      logic_justification:
        "Quantitative risk modeling shows concentration risk is offset by integrity gains.",
    },
  ],
};

// ============================================
// 2. DAGRE LAYOUT ENGINE
// ============================================
const NODE_W = 240;
const NODE_H = 90;

function getLayoutedElements(nodes, edges) {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: 'LR', ranksep: 280, nodesep: 50, edgesep: 20 });

  nodes.forEach((n) => g.setNode(n.id, { width: NODE_W, height: NODE_H }));
  edges.forEach((e) => g.setEdge(e.source, e.target));

  dagre.layout(g);

  return {
    nodes: nodes.map((n) => {
      const pos = g.node(n.id);
      return {
        ...n,
        position: { x: pos.x - NODE_W / 2, y: pos.y - NODE_H / 2 },
        style: { ...n.style, width: NODE_W, height: NODE_H },
      };
    }),
    edges,
  };
}

// ============================================
// 3. PARSER: JSON -> React Flow
// ============================================
function parseToReactFlow(data) {
  const traces = data.compiled_lineage_traces;
  const srcSet = [...new Set(traces.map((t) => t.source_node))];
  const tgtSet = [...new Set(traces.map((t) => t.target_concept))];

  const nodes = [];
  const edges = [];

  // Tier 1
  srcSet.forEach((label, i) => {
    nodes.push({
      id: `src-${i}`,
      type: 'tier1',
      data: { label },
      position: { x: 0, y: 0 },
    });
  });

  // Tier 2
  tgtSet.forEach((label, i) => {
    nodes.push({
      id: `tgt-${i}`,
      type: 'tier2',
      data: { label },
      position: { x: 0, y: 0 },
    });
  });

  // Tier 3
  nodes.push({
    id: 'roadmap',
    type: 'tier3',
    data: { label: 'Execution Roadmap' },
    position: { x: 0, y: 0 },
  });

  // T1 -> T2 edges
  traces.forEach((t, i) => {
    edges.push({
      id: `e-${i}`,
      source: `src-${srcSet.indexOf(t.source_node)}`,
      target: `tgt-${tgtSet.indexOf(t.target_concept)}`,
      type: 'custom',
      data: {
        relationship_type: t.relationship_type,
        extracted_fact: t.extracted_fact,
        logic_justification: t.logic_justification,
      },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#06b6d4' },
    });
  });

  // T2 -> T3 edges
  tgtSet.forEach((_, i) => {
    edges.push({
      id: `e-t3-${i}`,
      source: `tgt-${i}`,
      target: 'roadmap',
      type: 'smoothstep',
      style: { stroke: '#8b5cf6', strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#8b5cf6' },
      animated: true,
    });
  });

  return getLayoutedElements(nodes, edges);
}

// ============================================
// 4. CUSTOM NODE COMPONENTS
// ============================================
function Tier1Node({ data }) {
  return (
    <div className="group bg-slate-800/90 border border-cyan-500/30 rounded-xl p-4 w-[240px] shadow-lg shadow-cyan-900/10 hover:shadow-cyan-500/20 hover:border-cyan-400/60 transition-all duration-300 backdrop-blur-sm">
      <Handle
        type="source"
        position={Position.Right}
        className="!bg-cyan-400 !w-2.5 !h-2.5 !border-2 !border-slate-900"
      />
      <div className="flex items-center gap-2 mb-2">
        <span className="flex h-5 w-5 items-center justify-center rounded bg-cyan-500/10">
          <svg className="w-3 h-3 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
          </svg>
        </span>
        <span className="text-[10px] font-bold uppercase tracking-widest text-cyan-400/80">Tier 1 Anchor</span>
      </div>
      <div className="text-sm text-white font-semibold leading-snug">{data.label}</div>
    </div>
  );
}

function Tier2Node({ data }) {
  return (
    <div className="group bg-slate-800/90 border border-violet-500/30 rounded-xl p-4 w-[240px] shadow-lg shadow-violet-900/10 hover:shadow-violet-500/20 hover:border-violet-400/60 transition-all duration-300 backdrop-blur-sm">
      <Handle
        type="target"
        position={Position.Left}
        className="!bg-violet-400 !w-2.5 !h-2.5 !border-2 !border-slate-900"
      />
      <Handle
        type="source"
        position={Position.Right}
        className="!bg-violet-400 !w-2.5 !h-2.5 !border-2 !border-slate-900"
      />
      <div className="flex items-center gap-2 mb-2">
        <span className="flex h-5 w-5 items-center justify-center rounded bg-violet-500/10">
          <svg className="w-3 h-3 text-violet-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
          </svg>
        </span>
        <span className="text-[10px] font-bold uppercase tracking-widest text-violet-400/80">Tier 2 Chunk</span>
      </div>
      <div className="text-sm text-white font-semibold leading-snug">{data.label}</div>
    </div>
  );
}

function Tier3Node({ data }) {
  return (
    <div className="group relative bg-gradient-to-br from-indigo-950 via-slate-900 to-slate-950 border-2 border-indigo-400/40 rounded-2xl p-5 w-[260px] shadow-2xl shadow-indigo-900/30 hover:shadow-indigo-500/30 hover:border-indigo-300/60 transition-all duration-300 cursor-pointer">
      <div className="absolute -inset-0.5 bg-gradient-to-r from-indigo-500 to-cyan-500 rounded-2xl opacity-0 group-hover:opacity-20 blur transition duration-500" />
      <Handle
        type="target"
        position={Position.Left}
        className="!bg-indigo-400 !w-3 !h-3 !border-2 !border-slate-900"
      />
      <div className="relative">
        <div className="flex items-center gap-2 mb-3">
          <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-indigo-500/20">
            <svg className="w-4 h-4 text-indigo-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </span>
          <span className="text-[10px] font-bold uppercase tracking-widest text-indigo-300/80">Tier 3 Synthesis</span>
        </div>
        <div className="text-lg text-white font-bold leading-tight">{data.label}</div>
        <div className="mt-3 flex items-center gap-1.5 text-[11px] text-indigo-300/70">
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            <path strokeLinecap="round" strokeLinejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
          </svg>
          Click to reveal logic
        </div>
      </div>
    </div>
  );
}

// ============================================
// 5. MAIN APPLICATION
// ============================================
export default function ATISDashboard() {
  const [showPanel, setShowPanel] = useState(false);
  const [hoveredEdge, setHoveredEdge] = useState(null);
  const [selectedCountry, setSelectedCountry] = useState('Zimbabwe');

  // Memoize initial graph
  const { nodes: initialNodes, edges: initialEdges } = useMemo(
    () => parseToReactFlow(reasoningData),
    []
  );

  const [nodes, , onNodesChange] = useNodesState(initialNodes);
  const [edges, , onEdgesChange] = useEdgesState(initialEdges);

  // Custom edge defined inside component so it can access setHoveredEdge
  const CustomEdge = useMemo(() => {
    return ({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, data, markerEnd }) => {
      const [path, labelX, labelY] = getBezierPath({
        sourceX,
        sourceY,
        sourcePosition,
        targetX,
        targetY,
        targetPosition,
      });

      const isHovered = hoveredEdge?.id === id;

      return (
        <g
          onMouseEnter={() =>
            setHoveredEdge({
              id,
              x: labelX,
              y: labelY,
              data,
            })
          }
          onMouseLeave={() => setHoveredEdge(null)}
          style={{ cursor: 'pointer' }}
        >
          <path
            d={path}
            fill="none"
            stroke={isHovered ? '#22d3ee' : '#475569'}
            strokeWidth={isHovered ? 3 : 1.5}
            className="transition-all duration-300"
            markerEnd={markerEnd}
          />
        </g>
      );
    };
  }, [hoveredEdge]);

  const edgeTypes = useMemo(() => ({ custom: CustomEdge }), [CustomEdge]);
  const nodeTypes = useMemo(
    () => ({ tier1: Tier1Node, tier2: Tier2Node, tier3: Tier3Node }),
    []
  );

  const onNodeClick = useCallback((_, node) => {
    if (node.type === 'tier3') {
      setShowPanel(true);
    }
  }, []);

  const metrics = reasoningData.metrics;

  return (
    <div className="w-screen h-screen bg-slate-950 text-white overflow-hidden flex flex-col font-sans selection:bg-cyan-500/30">
      {/* ---------- ZONE A: Value Banner ---------- */}
      <header className="shrink-0 h-[72px] bg-slate-900/80 border-b border-slate-800 flex items-center justify-between px-6 backdrop-blur-md z-20">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 bg-cyan-500/10 rounded-lg flex items-center justify-center border border-cyan-500/20">
            <svg className="w-5 h-5 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
            </svg>
          </div>
          <div>
            <h1 className="text-base font-bold tracking-tight text-white">ATIS Intelligence Engine</h1>
            <p className="text-[11px] text-slate-400 font-medium">Pipeline Visualization Dashboard</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 text-[11px] font-bold uppercase tracking-widest text-slate-400">
            Analysing from
            <select
              value={selectedCountry}
              onChange={(event) => setSelectedCountry(event.target.value)}
              className="rounded-lg border border-slate-700 bg-slate-800 px-2 py-1.5 text-xs font-semibold normal-case tracking-normal text-white"
            >
              {Object.keys(PERSPECTIVE_COUNTRY_CODES).map((country) => (
                <option key={country} value={country}>{country}</option>
              ))}
            </select>
          </label>
          <MetricCard
            icon={
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            }
            label="Vault Files"
            value={metrics.total_vault_files_scanned.toLocaleString()}
          />
          <MetricCard
            icon={
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
              </svg>
            }
            label="Nodes Extracted"
            value={metrics.nodes_extracted}
          />
          <MetricCard
            icon={
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
            }
            label="Chunks Processed"
            value={metrics.map_chunks_processed.toLocaleString()}
          />
          <MetricCard
            icon={
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            }
            label="Hours Saved"
            value={`${metrics.estimated_manual_hours_saved}h`}
            highlight
          />
        </div>
      </header>

      {/* ---------- ZONE B: React Flow Canvas ---------- */}
      <main className="flex-1 relative">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={onNodeClick}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          fitView
          fitViewOptions={{ padding: 0.25, duration: 800 }}
          minZoom={0.3}
          maxZoom={1.5}
          defaultEdgeOptions={{ type: 'custom', animated: true }}
          proOptions={{ hideAttribution: true }}
        >
          <Background color="#1e293b" gap={24} size={1} className="bg-slate-950" />
          <Controls className="!bg-slate-800/90 !border-slate-700 !text-white !shadow-xl" />
          <MiniMap
            className="!bg-slate-900/90 !border-slate-700 !rounded-lg !shadow-xl"
            nodeColor={(n) => {
              if (n.type === 'tier1') return '#06b6d4';
              if (n.type === 'tier2') return '#8b5cf6';
              return '#6366f1';
            }}
            maskColor="rgba(15, 23, 42, 0.75)"
          />
        </ReactFlow>

        {/* Edge Hover Tooltip */}
        {hoveredEdge && (
          <div
            className="absolute z-50 pointer-events-none"
            style={{
              left: hoveredEdge.x,
              top: hoveredEdge.y,
              transform: 'translate(-50%, -110%)',
            }}
          >
            <div className="bg-slate-900/95 border border-cyan-500/40 rounded-xl p-4 shadow-2xl backdrop-blur-xl w-[320px]">
              <div className="flex items-center gap-2 mb-2">
                <span className="px-2 py-0.5 rounded bg-cyan-500/10 border border-cyan-500/20 text-[10px] font-bold uppercase tracking-wider text-cyan-300">
                  {hoveredEdge.data.relationship_type}
                </span>
              </div>
              <p className="text-sm text-white font-medium mb-2 leading-snug">
                {hoveredEdge.data.extracted_fact}
              </p>
              <div className="border-t border-slate-700 pt-2">
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  {hoveredEdge.data.logic_justification}
                </p>
              </div>
              <div className="absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-1/2 w-3 h-3 bg-slate-900 border-r border-b border-cyan-500/40 rotate-45" />
            </div>
          </div>
        )}

        {/* ---------- ZONE C: Synthesis Panel ---------- */}
        <aside
          className={`absolute top-0 right-0 h-full w-[440px] bg-slate-900/95 border-l border-slate-700/50 shadow-2xl backdrop-blur-xl transform transition-transform duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] z-30 ${
            showPanel ? 'translate-x-0' : 'translate-x-full'
          }`}
        >
          <div className="p-6 h-full flex flex-col">
            <div className="flex items-start justify-between mb-6">
              <div>
                <h2 className="text-xl font-bold text-white flex items-center gap-2.5">
                  <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-indigo-500/10">
                    <svg className="w-4 h-4 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                    </svg>
                  </span>
                  Execution Roadmap
                </h2>
                <p className="text-xs text-slate-400 mt-1 font-medium">Tier 3 Synthesis Logic</p>
              </div>
              <button
                onClick={() => setShowPanel(false)}
                className="p-2 hover:bg-slate-800 rounded-lg transition-colors border border-transparent hover:border-slate-700"
              >
                <svg className="w-5 h-5 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="flex-1 overflow-y-auto pr-1 space-y-6 custom-scrollbar">
              {/* Synthesis Block */}
              <div className="bg-slate-800/40 rounded-2xl p-5 border border-slate-700/40">
                <div className="text-[10px] font-bold text-indigo-400 uppercase tracking-widest mb-3 flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-indigo-400" />
                  Synthesis Output
                </div>
                <p className="text-[13px] text-slate-300 leading-relaxed">
                  {reasoningData.convergence_flow.tier_3_synthesis_logic}
                </p>
              </div>

              {/* Tier 1 List */}
              <div>
                <div className="text-[10px] font-bold text-cyan-400 uppercase tracking-widest mb-3 flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                  Tier 1 Anchors
                </div>
                <div className="space-y-2">
                  {reasoningData.convergence_flow.tier_1_anchors.map((anchor, i) => (
                    <div
                      key={i}
                      className="flex items-center gap-3 text-xs text-slate-300 bg-slate-800/30 p-3 rounded-xl border border-slate-700/30 hover:border-cyan-500/20 transition-colors"
                    >
                      <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-cyan-500/10 text-cyan-400 text-[10px] font-bold">
                        {i + 1}
                      </span>
                      {anchor}
                    </div>
                  ))}
                </div>
              </div>

              {/* Tier 2 List */}
              <div>
                <div className="text-[10px] font-bold text-violet-400 uppercase tracking-widest mb-3 flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-violet-400" />
                  Tier 2 Processing Chunks
                </div>
                <div className="space-y-2">
                  {reasoningData.convergence_flow.tier_2_processing_chunks.map((chunk, i) => (
                    <div
                      key={i}
                      className="flex items-center gap-3 text-xs text-slate-300 bg-slate-800/30 p-3 rounded-xl border border-slate-700/30 hover:border-violet-500/20 transition-colors"
                    >
                      <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-violet-500/10 text-violet-400 text-[10px] font-bold">
                        {i + 1}
                      </span>
                      {chunk}
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="mt-4 pt-4 border-t border-slate-800 flex items-center justify-between text-[11px] text-slate-500">
              <span>ATIS Engine v2.4</span>
              <span className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                System Active
              </span>
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}

// ============================================
// 6. HELPER COMPONENTS
// ============================================
function MetricCard({ icon, label, value, highlight }) {
  return (
    <div
      className={`flex items-center gap-3 px-3.5 py-2 rounded-xl border transition-colors ${
        highlight
          ? 'bg-cyan-500/5 border-cyan-500/15 hover:bg-cyan-500/10'
          : 'bg-slate-800/30 border-slate-700/20 hover:bg-slate-800/50'
      }`}
    >
      <div className={`${highlight ? 'text-cyan-400' : 'text-slate-400'}`}>{icon}</div>
      <div>
        <div className={`text-base font-bold leading-none ${highlight ? 'text-cyan-300' : 'text-white'}`}>
          {value}
        </div>
        <div className="text-[10px] text-slate-500 uppercase tracking-wider mt-1 font-medium">{label}</div>
      </div>
    </div>
  );
}
