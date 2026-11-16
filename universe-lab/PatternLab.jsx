import { useEffect, useRef, useState } from 'react';
import { bondColors } from './pattern-experiment';

function positionsFor(members, bonds, cx, cy) {
  if (members.length === 1) return { [members[0]]: { x: cx, y: cy } };
  if (members.length === 2) return {
    [members[0]]: { x: cx - 22, y: cy }, [members[1]]: { x: cx + 22, y: cy },
  };
  if (bonds.length === 2) {
    const center = members.find((id) => bonds.filter((bond) => bond.quark === id || bond.partner === id).length === 2);
    const ends = members.filter((id) => id !== center);
    return { [ends[0]]: { x: cx - 32, y: cy }, [center]: { x: cx, y: cy }, [ends[1]]: { x: cx + 32, y: cy } };
  }
  return { [members[0]]: { x: cx - 25, y: cy + 10 }, [members[1]]: { x: cx + 25, y: cy + 10 },
    [members[2]]: { x: cx, y: cy - 25 } };
}

export default function PatternLab({ events, bonds, groups }) {
  const mapRef = useRef(null);
  const [follow, setFollow] = useState(true);
  const quarks = events.filter((event) => event.stage === 'spark' || event.stage === 'reveal');
  const items = [{ members: [0], seeder: true }, ...groups];
  const width = 800;
  const height = Math.ceil(items.length / 4) * 150;
  const latestGroup = items.findIndex((group) => group.members.includes(quarks.length - 1));
  useEffect(() => {
    if (follow && mapRef.current && latestGroup >= 0) {
      mapRef.current.scrollLeft = (latestGroup % 4) * 200 + 100 - mapRef.current.clientWidth / 2;
      mapRef.current.scrollTop = Math.floor(latestGroup / 4) * 150 + 75 - mapRef.current.clientHeight / 2;
    }
  }, [follow, latestGroup, quarks.length, height]);
  return <figure className="universe-figure">
    <label className="universe-follow"><input type="checkbox" checked={follow}
      onChange={(event) => setFollow(event.target.checked)} /> Follow the newest quark</label>
    <div className="universe-map" ref={mapRef}>
      <svg className="pattern-experiment-graph" width={width} height={height} viewBox={`0 0 ${width} ${height}`}
        role="img" aria-label={`${groups.length} separate candidate and incomplete groups, plus the seeder`}>
        {items.map((group, index) => {
          const x = (index % 4) * 200;
          const y = Math.floor(index / 4) * 150;
          const groupBonds = bonds.filter((bond) => group.members.includes(bond.quark) && group.members.includes(bond.partner));
          const positions = positionsFor(group.members, groupBonds, x + 100, y + 78);
          const label = group.seeder ? 'Seeder' : group.complete
            ? `Candidate ${index} · ${groupBonds.length} bonds` : 'Not complete';
          return <g key={group.members.join('-')} data-pattern-group={group.seeder ? 'seeder' : group.complete ? 'candidate' : 'incomplete'}
            data-members={group.members.join(',')}>
            <rect x={x + 5} y={y + 5} width="190" height="140" rx="8"
              fill={group.complete ? '#142d3d' : '#17283e'} stroke="#43536c" />
            <text x={x + 100} y={y + 25} textAnchor="middle" fill="#bdcbe0" fontSize="12">{label}</text>
            {groupBonds.map((bond) => {
              const from = bond.outcome === 'incoming' ? bond.partner : bond.quark;
              const to = bond.outcome === 'incoming' ? bond.quark : bond.partner;
              const color = bond.outcome === 'both' ? 'both' : `${quarks[from].bit}${quarks[to].bit}`;
              const start = positions[from];
              const end = positions[to];
              const distance = Math.hypot(end.x - start.x, end.y - start.y);
              const dx = (end.x - start.x) / distance;
              const dy = (end.y - start.y) / distance;
              const strength = bond.outcome === 'both' ? bond.outgoing : Math.max(bond.outgoing, bond.incoming);
              return <line key={`${Math.min(from, to)}-${Math.max(from, to)}`}
                x1={start.x + dx * 10} y1={start.y + dy * 10} x2={end.x - dx * 10} y2={end.y - dy * 10}
                stroke={bondColors[color]} strokeWidth={2 * strength}>
                <title>{bond.outcome === 'both' ? `#${from + 1} ↔ #${to + 1}: strength 0.25 each`
                  : `#${from + 1} → #${to + 1}: strength 0.5`}</title>
              </line>;
            })}
            {group.members.map((id) => <g key={id}>
              <title>{id === 0 ? '#1: the original seeder' : `#${id + 1}: ${quarks[id].bit}`}</title>
              <circle cx={positions[id].x} cy={positions[id].y} r="9"
                fill={id === 0 ? '#b8622c' : quarks[id].bit ? '#15727b' : '#354a79'}
                stroke={id === 0 ? '#ffd28a' : quarks[id].bit ? '#78e5dc' : '#9daeff'} strokeWidth="1.5" />
              <text x={positions[id].x} y={positions[id].y + 3.5} textAnchor="middle" fill="#fff" fontSize="10">{quarks[id].bit}</text>
              <text x={positions[id].x} y={positions[id].y + 18} textAnchor="middle" fill="#bdcbe0" fontSize="8">#{id + 1}</text>
            </g>)}
          </g>;
        })}
      </svg>
    </div>
    <figcaption>{groups.filter((group) => group.complete).length} complete candidates;{' '}
      {groups.filter((group) => !group.complete).length} incomplete groups. Spacing is visual only, not physical repulsion.
      Candidates are unassigned and do not yet connect to other letters.</figcaption>
  </figure>;
}
