import { Component, type ReactNode } from 'react'

interface Props { children: ReactNode }
interface State { hasError: boolean; error: string | null }

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null }
  static getDerivedStateFromError(e: Error) { return { hasError: true, error: e.message } }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{ display:'flex',alignItems:'center',justifyContent:'center',height:'100vh',flexDirection:'column',gap:16,background:'#09090b',color:'#e5e2e1',fontFamily:'monospace',padding:24 }}>
          <h2 style={{color:'#00f2ff',fontSize:18}}>Something crashed</h2>
          <pre style={{fontSize:12,color:'#b9cacb',maxWidth:600,whiteSpace:'pre-wrap',textAlign:'center'}}>{this.state.error}</pre>
          <button onClick={() => { this.setState({hasError:false,error:null}); window.location.reload() }}
            style={{padding:'8px 24px',background:'#00f2ff',color:'#00363a',border:'none',cursor:'pointer',fontFamily:'monospace',fontWeight:600}}>
            Reload
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
